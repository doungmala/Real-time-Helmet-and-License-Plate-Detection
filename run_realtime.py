# detector/run_realtime.py
import os, time, cv2, requests
from typing import Optional, Tuple

# --- Helmet YOLOv3 ---
from detector.helmet import HelmetDetector

# --- Plate locate / number (Ultralytics .pt) ---
from detector.plate_locate_yolo import PlateLocateV8
from detector.plate_number_yolo import PlateNumberV8

# ===================== CONFIG =====================
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())   # จะไปหา .env ที่รากโปรเจกต์ให้อัตโนมัติ

BACKEND   = os.getenv("BACKEND_URL", "http://localhost:8000")
DISABLE_BACKEND_POST = bool(int(os.getenv("DISABLE_BACKEND_POST", "0")))
CAMERA_ID = os.getenv("CAMERA_ID", "KK1")
LOCATION  = os.getenv("LOCATION",  "ถ.มิตรภาพ (หน้าเซ็นทรัล)")

COOLDOWN_HELMET   = float(os.getenv("COOLDOWN_HELMET_SEC", "5.0"))
COOLDOWN_NOHELMET = float(os.getenv("COOLDOWN_NOHELMET_SEC", "8.0"))

READ_PLATE_MODE   = os.getenv("READ_PLATE_MODE", "all").lower()  # "all" / "nohelmet"
OCR_MODE          = os.getenv("OCR_MODE", "hybrid").lower()      # "api" | "local" | "hybrid"

VIDEO_SOURCE = os.getenv("SOURCE")
try:
    VIDEO_SOURCE = int(VIDEO_SOURCE)
except ValueError:
    pass

# path โมเดล .pt
PLATE_LOCATE_PT = os.getenv("PLATE_LOCATE_PT", "config_helmet/plate_locate_v8/best.pt")
PLATE_NUMBER_PT = os.getenv("PLATE_NUMBER_PT", "config_helmet/plate_number_v8/best.pt")

LOCATE_IMGSZ = int(os.getenv("LOCATE_IMGSZ", "960"))
LOCATE_CONF  = float(os.getenv("LOCATE_CONF", "0.10"))
LOCATE_IOU   = float(os.getenv("LOCATE_IOU",  "0.50"))

NUMBER_IMGSZ = int(os.getenv("NUMBER_IMGSZ", "320"))
NUMBER_CONF  = float(os.getenv("NUMBER_CONF", "0.25"))
NUMBER_IOU   = float(os.getenv("NUMBER_IOU",  "0.50"))
NUMBER_LABEL_MAP = os.getenv("NUMBER_LABEL_MAP", "")
NUMBER_ONLY_THAI = bool(int(os.getenv("NUMBER_ONLY_THAI", "1")))

DEBUG_DRAW_ALL  = False
SAVE_PLATE_CROP = False
PLATE_CROPS_DIR = os.getenv("PLATE_CROPS_DIR", "output/plates")
DEBUG_PLATE_CHARS = False
SHOW_PLATE_CROP = False
FAILED_PLATE_DIR = os.getenv("FAILED_PLATE_DIR", "output/plates_failed")

# API Key สำหรับ AIForThai
AIFORTHAI_KEY = os.getenv("AIFORTHAI_KEY", "")
# ==================================================

LAST_POST = {"helmet": 0.0, "nohelmet": 0.0, "plate_only": 0.0}

def post_event(has_helmet: bool, plate: Optional[str], conf: Optional[float]) -> None:
    """ส่งเหตุการณ์เข้า Backend"""
    payload = {
        "camera_id":   CAMERA_ID,
        "location":    LOCATION,
        "has_helmet":  bool(has_helmet),
        "plate_number": plate,
        "confidence":   conf,
    }
    if DISABLE_BACKEND_POST or not BACKEND:
        print("[EVENT] (skip post) ->", payload)
        return
    try:
        r = requests.post(f"{BACKEND}/events", json=payload, timeout=3)
        if r.status_code != 200:
            print("Backend error:", r.text)
        else:
            print("[EVENT] ->", payload)  # debug log
    except Exception as e:
        print("POST /events failed:", e)

def try_locate_plate(det_plate: PlateLocateV8, frame, bbox) -> Optional[Tuple[int,int,int,int,float]]:
    """ คืน (px1,py1,px2,py2, conf) หรือ None """
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    # ---------- ROI ท้ายรถ ----------
    ry1 = int(y1 + (y2 - y1) * 0.35)
    rx1 = max(0, x1 - 120); rx2 = min(W, x2 + 120)
    ry2 = min(H, y2 + 140)
    roi = frame[ry1:ry2, rx1:rx2]

    if roi.size:
        up = cv2.resize(roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        pb = det_plate.detect_best(up)
        if pb:
            px1, py1, px2, py2, pconf = pb
            px1 = int(px1 / 2) + rx1;  px2 = int(px2 / 2) + rx1
            py1 = int(py1 / 2) + ry1;  py2 = int(py2 / 2) + ry1
            return (px1, py1, px2, py2, pconf)

    # ---------- fallback: ทั้งเฟรม ----------
    big = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    pb2 = det_plate.detect_best(big)
    if pb2:
        qx1, qy1, qx2, qy2, qconf = pb2
        qx1 = int(qx1 / 1.5); qx2 = int(qx2 / 1.5)
        qy1 = int(qy1 / 1.5); qy2 = int(qy2 / 1.5)
        return (qx1, qy1, qx2, qy2, qconf)

    return None

def read_plate_with_api(crop) -> Tuple[Optional[str], Optional[float]]:
    """ เรียก AIForThai API อ่านป้าย """
    if not AIFORTHAI_KEY:
        return None, None
    ok, buf = cv2.imencode(".jpg", crop)
    if not ok:
        return None, None

    files = {"image": ("plate.jpg", buf.tobytes(), "image/jpeg")}
    payload = {"crop": "1", "rotate": "0"}
    headers = {"Apikey": AIFORTHAI_KEY}

    try:
        r = requests.post("https://api.aiforthai.in.th/lpr-v2",
                          headers=headers, data=payload, files=files, timeout=5)
        if r.status_code != 200:
            print("AIForThai error:", r.text)
            return None, None

        data = r.json()
        if "result" in data and data["result"]:
            plate = data["result"][0].get("license_plate", "")
            conf  = data["result"][0].get("confidence", 0)
            return plate, conf
    except Exception as e:
        print("AIForThai request failed:", e)
    return None, None

def main():
    # 1) Helmet detector
    det_helmet = HelmetDetector(
        cfg="config_helmet/yolov3-obj.cfg",
        weights="config_helmet/yolov3-obj_2400.weights",
        names="config_helmet/obj.names",
        input_size=(416, 416),
    )

    # 2) Plate locate / number
    det_plate = PlateLocateV8(
        weights=PLATE_LOCATE_PT, input_size=LOCATE_IMGSZ,
        conf_th=LOCATE_CONF, iou_th=LOCATE_IOU
    )
    det_text  = PlateNumberV8(
        weights=PLATE_NUMBER_PT, input_size=NUMBER_IMGSZ,
        conf_th=NUMBER_CONF, iou_th=NUMBER_IOU,
        preprocess=True,
        label_map_path=NUMBER_LABEL_MAP if NUMBER_LABEL_MAP else None,
        only_thai_digits=NUMBER_ONLY_THAI
    )

    cap = cv2.VideoCapture(VIDEO_SOURCE)
   
    if not cap.isOpened():
        raise RuntimeError(f"เปิดวิดีโอ/กล้องไม่ได้: {VIDEO_SOURCE}")


    if SAVE_PLATE_CROP:
        os.makedirs(PLATE_CROPS_DIR, exist_ok=True)
        

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()
        riders = det_helmet.detect(frame)

        # --- Always run plate detection on the full frame (for every car, every frame) ---
        fullframe_plate_text, fullframe_avg_conf = None, None
        plate_bbox = det_plate.detect_best(frame)
        if plate_bbox:
            px1, py1, px2, py2, pconf = plate_bbox
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 255), 2)

            # Expand the crop a bit for better OCR
            ph = py2 - py1
            pw = px2 - px1
            pad_y = max(2, int(0.08 * ph))
            pad_x = max(2, int(0.08 * pw))
            cy1 = max(0, py1 - pad_y)
            cy2 = min(frame.shape[0], py2 + pad_y)
            cx1 = max(0, px1 - pad_x)
            cx2 = min(frame.shape[1], px2 + pad_x)
            crop = frame[cy1:cy2, cx1:cx2]

            if SHOW_PLATE_CROP:
                try:
                    cv2.imshow("Plate Crop", crop)
                except Exception:
                    pass

            if SAVE_PLATE_CROP:
                ts = int(time.time() * 1000)
                cv2.imwrite(os.path.join(PLATE_CROPS_DIR, f"plate_{ts}.jpg"), crop)

            # --------- OCR ตาม OCR_MODE ----------
            if OCR_MODE == "api":
                plate_text, avg_conf = read_plate_with_api(crop)
                if not plate_text:
                    text, boxes, avg_conf = det_text.read_text(crop)
                    plate_text = text
            elif OCR_MODE == "local":
                text, boxes, avg_conf = det_text.read_text(crop)
                plate_text = text
            else:  # hybrid
                text, boxes, avg_conf = det_text.read_text(crop)
                plate_text = text
                if not text:
                    plate_text, avg_conf = read_plate_with_api(crop)
            # ------------------------------------

            if plate_text:
                cv2.putText(frame, plate_text, (px1, max(0, py1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                fullframe_plate_text = plate_text
                fullframe_avg_conf = avg_conf
            else:
                try:
                    os.makedirs(FAILED_PLATE_DIR, exist_ok=True)
                    ts = int(time.time() * 1000)
                    cv2.imwrite(os.path.join(FAILED_PLATE_DIR, f"failed_{ts}.jpg"), crop)
                except Exception:
                    pass

            # Draw character boxes (debug)
            if DEBUG_PLATE_CHARS and 'boxes' in locals() and boxes:
                for bx in boxes:
                    bx1, by1, bx2, by2, clabel, cconf = bx
                    cv2.rectangle(frame, (cx1 + bx1, cy1 + by1), (cx1 + bx2, cy1 + by2), (0, 255, 255), 1)
                    cv2.putText(frame, str(clabel), (cx1 + bx1, max(0, cy1 + by1 - 2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # --- Continue with helmet/person detection as before ---
        for r in riders:
            (x1, y1, x2, y2) = r["bbox"]
            has_helmet = bool(r["has_helmet"])
            conf_h     = float(r["conf"])

            color = (0, 200, 0) if has_helmet else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{'Helmet' if has_helmet else 'No-Helmet'} {conf_h:.2f}",
                        (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            should_read = (READ_PLATE_MODE == "all") or ((READ_PLATE_MODE == "nohelmet") and (not has_helmet))

            plate_text, avg_conf = None, None
            plate_bbox = None

            if should_read:
                plate_bbox = try_locate_plate(det_plate, frame, (x1, y1, x2, y2))
                if plate_bbox:
                    px1, py1, px2, py2, pconf = plate_bbox
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2)

                    # ขยายกรอบเล็กน้อยเพื่อเผื่อขอบตัวอักษร
                    ph = py2 - py1
                    pw = px2 - px1
                    pad_y = max(2, int(0.08 * ph))
                    pad_x = max(2, int(0.08 * pw))
                    cy1 = max(0, py1 - pad_y)
                    cy2 = min(frame.shape[0], py2 + pad_y)
                    cx1 = max(0, px1 - pad_x)
                    cx2 = min(frame.shape[1], px2 + pad_x)
                    crop = frame[cy1:cy2, cx1:cx2]

                    if SHOW_PLATE_CROP:
                        try:
                            cv2.imshow("Plate Crop", crop)
                        except Exception:
                            pass

                    if SAVE_PLATE_CROP:
                        ts = int(time.time() * 1000)
                        cv2.imwrite(os.path.join(PLATE_CROPS_DIR, f"plate_{ts}.jpg"), crop)

                    # --------- OCR ตาม OCR_MODE ----------
                    if OCR_MODE == "api":
                        plate_text, avg_conf = read_plate_with_api(crop)
                        # ถ้า API ไม่ได้ผล ให้ลองโมเดลโลคอลของลูกค้าต่อทันที
                        if not plate_text:
                            text, boxes, avg_conf = det_text.read_text(crop)
                            plate_text = text

                    elif OCR_MODE == "local":
                        text, boxes, avg_conf = det_text.read_text(crop)
                        plate_text = text

                    else:  # hybrid
                        text, boxes, avg_conf = det_text.read_text(crop)
                        plate_text = text
                        if not text:
                            plate_text, avg_conf = read_plate_with_api(crop)
                    # ------------------------------------

                    if plate_text:
                        cv2.putText(frame, plate_text, (px1, max(0, py1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    else:
                        # เก็บเคสที่อ่านไม่ได้เพื่อปรับจูนภายหลัง
                        try:
                            os.makedirs(FAILED_PLATE_DIR, exist_ok=True)
                            ts = int(time.time() * 1000)
                            cv2.imwrite(os.path.join(FAILED_PLATE_DIR, f"failed_{ts}.jpg"), crop)
                        except Exception:
                            pass

                    # วาดกรอบอักษรย่อย (debug)
                    if DEBUG_PLATE_CHARS and 'boxes' in locals() and boxes:
                        # กล่องอักษรเป็นพิกัดภายใน crop -> แปลงกลับสู่ภาพเต็มด้วย cx1,cy1 ของ crop ที่คำนวณไว้ด้านบน
                        for bx in boxes:
                            bx1, by1, bx2, by2, clabel, cconf = bx
                            cv2.rectangle(frame, (cx1 + bx1, cy1 + by1), (cx1 + bx2, cy1 + by2), (0, 255, 255), 1)
                            cv2.putText(frame, str(clabel), (cx1 + bx1, max(0, cy1 + by1 - 2)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # --- ส่งอีเวนต์ (คุมความถี่ด้วย cooldown) ---
            if has_helmet:
                if now - LAST_POST["helmet"] > COOLDOWN_HELMET:
                    post_event(True, plate_text, avg_conf or conf_h)
                    LAST_POST["helmet"] = now
            else:
                if now - LAST_POST["nohelmet"] > COOLDOWN_NOHELMET:
                    post_event(False, plate_text, avg_conf or conf_h)
                    LAST_POST["nohelmet"] = now

            if DEBUG_DRAW_ALL and plate_bbox is None:
                cv2.putText(frame, "Plate: not found", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # If we saw a plate but no riders at all, post a plate-only event (as no-helmet)
        if (not riders) and fullframe_plate_text:
            if now - LAST_POST["plate_only"] > COOLDOWN_NOHELMET:
                post_event(False, fullframe_plate_text, fullframe_avg_conf)
                LAST_POST["plate_only"] = now

        cv2.imshow("Helmet + Plate Realtime", frame)
        if cv2.waitKey(5) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
