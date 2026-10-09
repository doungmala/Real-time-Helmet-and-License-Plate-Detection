from ultralytics import YOLO

# ปรับให้ครอบคลุมชื่อคลาสของโมเดลลูกค้า
# detector/plate_locate_yolo.py
_PLATE_LABELS = {
  "plate", "license_plate", "lp", "license",
  "platelocate", "reg_plate", "licenseplate",
  "License_Plate"  # ← ต้องมีชื่อนี้
}


DEBUG_LOCATE = True  # เปิด debug ดูผลลัพธ์

class PlateLocateV8:
    def __init__(self, weights="config_helmet/plate_locate_v8/best.pt",
                 input_size=640, conf_th=0.35, iou_th=0.45):
        print(f"[Locate] loading weights: {weights}")
        self.model = YOLO(weights)
        self.input_size = input_size
        self.conf_th = conf_th
        self.iou_th = iou_th
        self.names = self.model.names  # id -> label
        print("[Locate] class names:", self.names)

    def detect_best(self, frame):
        """
        return: (x1,y1,x2,y2, conf) ของกรอบป้ายที่มั่นใจที่สุด หรือ None
        """
        res = self.model(frame, imgsz=self.input_size,
                         conf=self.conf_th, iou=self.iou_th, verbose=False)[0]
        best = None
        boxes_dbg = []   # ✅ กำหนด list ก่อน

        for b in res.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            conf = float(b.conf[0])
            cls  = int(b.cls[0])
            label = self.names.get(cls, str(cls)).lower()

            boxes_dbg.append((label, conf, (x1, y1, x2, y2)))  # เก็บไว้ debug

            if label in _PLATE_LABELS:
                if best is None or conf > best[-1]:
                    best = (x1, y1, x2, y2, conf)

        # if DEBUG_LOCATE:
        #     print("[Locate] raw boxes:", boxes_dbg)

        return best
