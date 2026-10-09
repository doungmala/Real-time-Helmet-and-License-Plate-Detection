# detector/plate.py  (ใช้โมเดลลูกค้า YOLOv8 แทน API เดิม)
import os
from typing import Optional, Tuple, List
import cv2
import numpy as np

# แก้ path ได้ตามที่คุณวางไฟล์ไว้
LOCATE_MODEL = os.getenv("PLATE_LOCATE_MODEL", "config_helmet/plate_locate_v8/best.pt")
NUMBER_MODEL = os.getenv("PLATE_NUMBER_MODEL", "config_helmet/plate_number_v8/best.pt")

# ค่ากรองเริ่มต้น (ปรับได้ตามความแม่นของโมเดลลูกค้า)
CONF_LOCATE = float(os.getenv("CONF_LOCATE", "0.45"))
CONF_NUMBER = float(os.getenv("CONF_NUMBER", "0.25"))

# -------- โหลดโมเดลครั้งเดียว --------
_ultra = None
_locate = None
_number = None
def _load_models():
    global _ultra, _locate, _number
    if _ultra is None:
        # โหลดแบบ lazy เพื่อลดเวลา import
        from ultralytics import YOLO
        _ultra = YOLO  # เก็บคลาสไว้
        _locate = _ultra(LOCATE_MODEL)
        _number = _ultra(NUMBER_MODEL)

def _resize_max(img, maxw=1000):
    h, w = img.shape[:2]
    if w <= maxw:
        return img
    scale = maxw / float(w)
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

def _read_chars_with_detector(crop_bgr, conf_th=CONF_NUMBER) -> str:
    """
    กรณีโมเดล NUMBER เป็น 'detector ของตัวอักษร' (พบได้บ่อยใน YOLOv8)
    - ดึง bbox ตัวอักษรทั้งหมด
    - sort ตามแกน x แล้วต่อเป็นสตริง
    """
    res = _number.predict(source=crop_bgr, conf=conf_th, verbose=False)
    if not res:
        return ""
    r = res[0]
    if r.boxes is None or r.boxes.xyxy is None:
        return ""
    names = r.names
    chars: List[tuple[int,str]] = []
    for box, cls_id in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
        x1,y1,x2,y2 = box.astype(int)
        label = names.get(int(cls_id), str(int(cls_id)))
        # กัน class แปลก ๆ เช่น 'plate' ที่อาจติดมาจาก training
        if len(label) == 1 and (label.isalnum() or label in ["ก","ข","ค","ง","จ","ฉ","ช","ซ","ญ","ฎ","ฏ","ฐ","ฑ","ฒ","ณ","ด","ต","ถ","ท","ธ","น","บ","ป","ผ","ฝ","พ","ฟ","ภ","ม","ย","ร","ฤ","ล","ว","ศ","ษ","ส","ห","ฬ","อ","ฮ"]):
            chars.append((int((x1+x2)//2), label))
    chars.sort(key=lambda t: t[0])
    return "".join([c for _, c in chars])

def read_plate_from_frame(frame_bgr, roi: Optional[Tuple[int,int,int,int]] = None):
    """
    คืนค่า: (plate_text, bbox_fullframe, conf_est)
    - ถ้าใช้ ROI จะหาป้ายในกรอบนั้นแล้วแปลง bbox กลับพิกัดภาพเต็มให้
    """
    _load_models()

    # 1) เตรียมภาพ (ทั้งเฟรมหรือ ROI)
    if roi:
        x1,y1,x2,y2 = map(int, roi)
        img = frame_bgr[max(0,y1):max(0,y2), max(0,x1):max(0,x2)].copy()
    else:
        img = frame_bgr
        x1=y1=0

    img_small = _resize_max(img, 1200)

    # 2) หา bbox ของป้ายด้วยโมเดล locate
    loc_res = _locate.predict(source=img_small, conf=CONF_LOCATE, verbose=False)
    if not loc_res:
        return None, None, None
    r = loc_res[0]
    if r.boxes is None or r.boxes.xyxy is None or len(r.boxes) == 0:
        return None, None, None

    # เลือกกล่องที่ conf สูงสุด (หรือจะวนทุกกล่องแล้วคืนตัวแรกก็ได้)
    best_i = int(np.argmax(r.boxes.conf.cpu().numpy()))
    bx = r.boxes.xyxy[best_i].cpu().numpy().astype(int).tolist()
    conf_loc = float(r.boxes.conf[best_i].cpu().numpy())

    # ปรับสเกลกลับ ถ้าภาพโดนย่อก่อน detect
    h0,w0 = img.shape[:2]
    hs,ws = img_small.shape[:2]
    sx, sy = w0/float(ws), h0/float(hs)
    xLT, yLT, xRB, yRB = int(bx[0]*sx), int(bx[1]*sy), int(bx[2]*sx), int(bx[3]*sy)

    crop = img[max(0,yLT):max(0,yRB), max(0,xLT):max(0,xRB)].copy()
    if crop.size == 0:
        return None, None, None

    # 3) อ่านตัวอักษรด้วยโมเดล number (สมมติเป็น detector ของตัวอักษร)
    text = _read_chars_with_detector(crop, conf_th=CONF_NUMBER).strip()

    # (ถ้าลูกค้าให้โมเดล NUMBER ที่เป็น OCR classifier/CRNN ให้สลับไปใช้ฟังก์ชันอื่นแทน _read_chars_with_detector)
    # ตัวอย่าง:
    # text = _read_with_ocr_classifier(crop)

    # 4) คืนค่าเป็นพิกัด "ภาพเต็ม"
    if roi:
        bbox_full = (xLT + x1, yLT + y1, xRB + x1, yRB + y1)
    else:
        bbox_full = (xLT, yLT, xRB, yRB)

    # conf ประเมินจาก locate (เพราะ number เป็นหลายตัวอักษร)
    return (text or None), bbox_full, conf_loc
