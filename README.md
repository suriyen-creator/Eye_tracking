# Eye_tracking


---

````markdown
# 👁️ Gaze + Object Tracking & Recognition

โปรเจ็กต์นี้ทำให้คอมพิวเตอร์สามารถ **ตรวจจับและติดตามวัตถุ** ในภาพ พร้อมทั้งใช้ **Eye/Gaze Tracking** เพื่อตรวจสอบว่าผู้ใช้กำลังมองวัตถุใดอยู่ และบอกได้ว่าวัตถุนั้นคืออะไร  

- รองรับการตรวจจับด้วย **YOLOv8**  
- รองรับการติดตาม (tracking) ด้วย **DeepSORT/ByteTrack**  
- ใช้ **MediaPipe FaceMesh + Iris** เพื่อหาตำแหน่งตาดำและคำนวณ gaze  
- สามารถ **คาลิเบรต 9 จุด** เพื่อความแม่นยำของ gaze-to-screen mapping  

---

## 🔧 Requirements

- **Python**: 3.9 – 3.11 (แนะนำ Python 3.10)  
- ระบบปฏิบัติการ: Windows / Linux / macOS  
- **Dependencies** (ติดตั้งผ่าน `pip`):
  - `opencv-python`
  - `mediapipe`
  - `ultralytics`
  - `numpy`
  - `scikit-learn`
  - `filterpy`
  - `joblib`

> ติดตั้งทั้งหมดได้จากไฟล์ `requirements.txt`

---

## 📦 Installation

1. สร้าง Virtual Environment  
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux / macOS
   venv\Scripts\activate      # Windows
````

2. ติดตั้ง dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. ดาวน์โหลด weight ของ YOLOv8 (Ultralytics จะโหลดอัตโนมัติเมื่อรันครั้งแรก)

---

## 🎯 Calibration (คาลิเบรตสายตา)

เพื่อให้ระบบรู้ว่าตาผู้ใช้กำลังมองตรงไหนบนหน้าจอ ต้องทำการคาลิเบรต:

```bash
python -m gaze.calibrate
```

* จะมีจุดขึ้นมา 9 จุด (3×3 grid) บนหน้าจอ
* มองจุดแต่ละจุดตามที่ขึ้นมา 0.5–1 วินาที
* เมื่อเสร็จสิ้น ระบบจะบันทึกโมเดล mapping ลงที่ `data/gaze_mapper.joblib`

---

## ▶️ Run Demo

หลังจากคาลิเบรตแล้ว ให้รันโปรแกรมหลัก:

```bash
python main.py
```

สิ่งที่จะเกิดขึ้น:

* กล้องจะแสดงภาพสด
* YOLO ตรวจจับวัตถุและแสดง bounding box
* จุดวงกลมสีเหลือง = ตำแหน่ง gaze ที่คำนวณได้
* กรอบวัตถุจะเปลี่ยนสีเมื่อคุณมอง
* ถ้ามองค้างที่วัตถุ ≥300ms ระบบจะขึ้นข้อความ **“คุณกำลังมอง: \[label]”**

---

## 📂 Project Structure

```
project/
├─ requirements.txt
├─ main.py               # loop หลัก
├─ gaze/
│  ├─ calibrate.py       # script คาลิเบรต 9 จุด
│  ├─ mapper.py          # โหลด/พยากรณ์ gaze mapping
├─ vision/
│  ├─ detector.py        # YOLO detector
│  ├─ tracker.py         # Object tracker (DeepSORT/ByteTrack)
│  └─ associate.py       # จับคู่ gaze ↔ object
└─ data/
   ├─ calib_samples.csv
   └─ gaze_mapper.joblib
```

---

## ⚠️ Known Issues & Tips

* **ความแม่นยำ** ขึ้นอยู่กับคุณภาพกล้อง, แสงสว่าง, และการคาลิเบรต
* ถ้าสวมแว่น → แสงสะท้อนอาจทำให้ iris detection เพี้ยน
* FPS อาจต่ำหากใช้ CPU → แนะนำให้ใช้ GPU (CUDA) สำหรับ YOLO
* mapping ของ gaze เป็น **per-user** ต้องคาลิเบรตใหม่หากเปลี่ยนผู้ใช้

---

## 🚀 Future Work

* รองรับ World-based gaze mapping (AR glasses / external camera)
* ต่อกับ **BLIP/CLIP** เพื่อให้คำอธิบายวัตถุเป็นประโยค
* เพิ่ม **Text-to-Speech (TTS)** เพื่ออ่านออกเสียงสิ่งที่ผู้ใช้มอง


