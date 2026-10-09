from ultralytics import YOLO
import numpy as np
import re
import os
import json

# แผนที่ตัวอักษรไทยตามลำดับ (ประมาณการ): A1 -> 'ก', A2 -> 'ข', ...
THAI_LETTERS = [
    'ก','ข','ฃ','ค','ฅ','ฆ','ง','จ','ฉ','ช','ซ','ฌ','ญ','ฎ','ฏ','ฐ','ฑ','ฒ','ณ',
    'ด','ต','ถ','ท','ธ','น','บ','ป','ผ','ฝ','พ','ฟ','ภ','ม','ย','ร','ฤ','ล','ว','ศ','ษ','ส','ห','ฬ','อ','ฮ'
]

# อนุญาตเฉพาะตัวอักษร/ตัวเลขที่มีโอกาสอยู่บนป้าย (ไทย, อังกฤษ, ตัวเลข, ขีด)
def _is_thai(ch: str) -> bool:
    return '\u0E00' <= ch <= '\u0E7F'

def _is_valid_char(ch: str, only_thai_digits: bool = False) -> bool:
    if not ch or len(ch) != 1:
        return False
    # ไทยช่วงยูนิโค้ด
    if _is_thai(ch):
        return True
    # จำกัดเฉพาะตัวเลขและสัญลักษณ์บางตัวเท่านั้น ถ้าบังคับโหมดไทยอย่างเดียว
    if only_thai_digits:
        if ch.isdigit() or ch in {'-', '·', '.', '_'}:
            return True
        return False
    # อังกฤษ/ตัวเลข/ขีดกลาง/ช่องว่างบางกรณี (โหมดทั่วไป)
    if ch.isalnum() or ch in {'-', '·', '.', '_'}:
        return True
    return False

def _normalize_label(label: str, label_map: dict | None = None) -> str:
    """ดึงตัวอักษรจริงจากชื่อคลาส เช่น 'th_ก' -> 'ก', 'digit_1' -> '1'"""
    if label is None:
        return ''
    s = str(label)
    s_stripped = s.strip()
    s_up = s_stripped.upper()

    # 1) ใช้ mapping ที่ผู้ใช้กำหนดถ้ามี (ลองหลายรูปแบบของ key)
    if label_map:
        for key in (s, s_stripped, s_up, s_stripped.lower(), re.sub(r"\s+", "", s_stripped), re.sub(r"\s+", "", s_up)):
            if key in label_map:
                return str(label_map[key])

    # จับ pattern เฉพาะที่พบบ่อยในโมเดลลูกค้า
    # N0..N9 -> ตัวเลข, A1..A44 -> ตัวอักษรไทยตามลำดับด้านบน
    mN = re.search(r'[Nn]\s*([0-9])', s_stripped)
    if mN:
        return mN.group(1)
    # ยอมรับรูปแบบ A-21, A_21, A 21, A021, a21 ฯลฯ
    mA = re.search(r'[Aa]\s*[-_]*\s*0*([0-9]{1,2})', s_stripped)
    if mA:
        idx = int(mA.group(1)) - 1
        if 0 <= idx < len(THAI_LETTERS):
            return THAI_LETTERS[idx]
        return ''

    # ค้นหาอักษรไทยที่อยู่ในชื่อคลาส ถ้ามีให้ใช้ก่อน (เช่น 'th_ถ', 'thai-ถ')
    m_th = re.findall(r'[\u0E00-\u0E7F]', s)
    if m_th:
        # ใช้ตัวสุดท้ายในสตริง (ส่วนใหญ่เป็นอักขระเป้าหมาย)
        return m_th[-1]

    # ตัด prefix ที่เป็น pattern ทั่วไป แล้วดึงตัวท้ายเป็น fallback สุดท้าย
    # ตัวอย่าง: 'digit_1', 'char_A', 'en_A'
    m = re.search(r'([A-Za-z0-9])$', s_stripped)
    if m:
        ch = m.group(1)
        # ถ้าเป็นอังกฤษหลายตัวติดกัน (เช่น BKK, CHON, UDON) ให้ “มองข้าม” เพราะเป็นจังหวัด ไม่ใช่อักขระป้ายหลัก
        if len(s_up) >= 3 and s_up.isalpha():
            return ''
        return ch
    # ถ้า label เป็นตัวเดียวอยู่แล้ว
    if len(s) == 1:
        return s
    return ''

class PlateNumberV8:
    def __init__(self, weights="config_helmet/plate_number_v8/best.pt",
                 input_size=320, conf_th=0.15, iou_th=0.5, preprocess: bool = True,
                 label_map_path: str | None = None, only_thai_digits: bool = True):
        self.model = YOLO(weights)
        self.input_size = input_size
        self.conf_th = conf_th
        self.iou_th = iou_th
        self.names = self.model.names  # id -> label str (เช่น 'ก','1',...)
        self.preprocess = preprocess
        self.only_thai_digits = only_thai_digits
        # debug mapping
        try:
            self.debug = bool(int(os.getenv("NUMBER_DEBUG", "0")))
        except Exception:
            self.debug = False
        # โหลด mapping ถ้ามี
        self.label_map = None
        try:
            if label_map_path and os.path.exists(label_map_path):
                with open(label_map_path, 'r', encoding='utf-8') as f:
                    self.label_map = json.load(f)
                print(f"[Number] loaded label map: {label_map_path} ({len(self.label_map)} entries)")
        except Exception as e:
            print("[Number] load label map failed:", e)
        try:
            print("[Number] class names:", self.names)
        except Exception:
            pass

    def read_text(self, plate_img):
        """
        return: (text, boxes, avg_conf)
        boxes: list of (x1,y1,x2,y2,label,conf)
        """
        # เลือกปรับภาพก่อนเข้าโมเดลเพื่อช่วยดึงตัวอักษรเล็ก/คอนทราสต์ต่ำ
        inp = plate_img
        try:
            import cv2
            if self.preprocess and inp is not None and hasattr(inp, 'shape'):
                h, w = inp.shape[:2]
                # upscale ถ้าเล็กเกินไป
                if h < 120:
                    scale = max(1.5, 160.0 / max(1, h))
                    nh, nw = int(h * scale), int(w * scale)
                    inp = cv2.resize(inp, (nw, nh), interpolation=cv2.INTER_CUBIC)
                # CLAHE บน Y channel
                yuv = cv2.cvtColor(inp, cv2.COLOR_BGR2YUV)
                y, u, v = cv2.split(yuv)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                y2 = clahe.apply(y)
                inp = cv2.cvtColor(cv2.merge((y2, u, v)), cv2.COLOR_YUV2BGR)
        except Exception:
            pass

        res = self.model(inp, imgsz=self.input_size,
                         conf=self.conf_th, iou=self.iou_th, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None, [], None
        chars = []
        for b in res.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            conf = float(b.conf[0])
            cls = int(b.cls[0])
            raw_label = self.names.get(cls, str(cls))
            label = _normalize_label(raw_label, self.label_map)
            if self.debug:
                try:
                    print(f"[NUMDBG] raw='{raw_label}' -> '{label}', conf={conf:.2f}")
                except Exception:
                    pass
            if not _is_valid_char(label, self.only_thai_digits):
                continue
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            chars.append((cy, cx, x1, y1, x2, y2, label, conf))

        if not chars:
            return None, [], None

        ys = np.array([c[0] for c in chars])
        # ถ้าแผงบน-ล่างห่างกันชัดเจน ใช้ median split, ไม่งั้นถือว่าเป็นบรรทัดเดียว
        if ys.max() - ys.min() > 12:  # พารามิเตอร์ขึ้นกับขนาดรูป/ฟอนต์
            mid = np.median(ys)
            line1 = [c for c in chars if c[0] <= mid]
            line2 = [c for c in chars if c[0] >  mid]
        else:
            line1, line2 = chars, []

        line1.sort(key=lambda c: c[1])
        line2.sort(key=lambda c: c[1])

        def to_str(line):
            s = "".join([c[6] for c in line])
            # ล้างช่องว่างซ้ำ ๆ
            return re.sub(r"\s+", " ", s).strip()

        text = to_str(line1)
        if line2:
            text = text + " " + to_str(line2)

        # กรองผลลัพธ์ให้เหลือเฉพาะไทย+ตัวเลข (หากเปิดโหมดนี้)
        if self.only_thai_digits and text:
            filtered = []
            for ch in text:
                if _is_thai(ch) or ch.isdigit() or ch in {'-', ' '}:
                    filtered.append(ch)
            text = ''.join(filtered)

        boxes = [(c[2], c[3], c[4], c[5], c[6], c[7]) for c in (line1 + line2)]
        avg_conf = float(np.mean([c[7] for c in (line1 + line2)])) if (line1 or line2) else None
        return text, boxes, avg_conf
