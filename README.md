# MIZHI: Advanced CCTV-Based Weapon & Threat Detection System

> **MIZHI** is an AI-powered, GUI-based application for real-time detection of weapons, suspicious objects, and aggressive human behavior from CCTV feeds or video files.  
> It combines **YOLOv8 object detection**, **MediaPipe pose estimation**, and **LSTM-based behavioral analysis** to provide instant alerts for enhanced security monitoring.

---

## 📌 Features

- **Real-Time Object Detection** using YOLOv8
- **Weapon Classification**:
  - 🟥 **Red Box** – Real weapons (gun, knife, rifle, sword, axe, etc.)
  - 🟨 **Yellow Box** – Potential fake weapon sources (mobile, laptop, TV, etc.)
  - 🟧 **Orange Box** – Suspicious objects (stick, tool, wrench, screwdriver)
- **Pose Detection & Aggression Analysis** using MediaPipe
- **Behavioral Analysis** using ResNet50 feature extractor + LSTM
- **Automated Alerts** overlay on live video
- **Multi-source Input** (camera feed or video file)
- **Model Training Interface** for custom datasets
- **Simple GUI** built with Tkinter

---

## 🛠️ Tech Stack

| Component         | Technology |
|-------------------|------------|
| Object Detection  | YOLOv8 (Ultralytics) |
| Pose Estimation   | MediaPipe Pose |
| Behavioral Model  | ResNet50 + LSTM |
| GUI               | Tkinter |
| Backend           | OpenCV, TensorFlow/Keras |
| Language          | Python 3.x |

---

## 📂 Project Structure

```

MIZHI/
│
├── mizhi\_detector.py         # Main application file
├── models/                   # Trained models (.h5 files)
│   ├── mizhi\_cnn\_final.h5
│   ├── mizhi\_lstm\_final.h5
│   └── mizhi\_feature\_extractor.h5
├── assets/                   # Images, architecture diagrams (optional)
├── requirements.txt          # Dependencies list
└── README.md                 # Documentation

````

---

## 🚀 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/anzilmthajudheenMIZHI-ADVANCED-THREAT-DETECTION-SYSTEM.git
cd MIZHI-ADVANCED-THREAT-DETECTION-SYSTEM
````

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

**Example `requirements.txt`**

```
opencv-python
numpy
mediapipe
tensorflow
ultralytics
scikit-learn
Pillow
```

### 3️⃣ Download YOLOv8 Model Weights

```bash
yolo download model=yolov8m.pt
```

Or manually place `yolov8m.pt` in the project directory.

### 4️⃣ Run the Application

```bash
python mizhi_detector.py
```

---

## 🎮 How to Use

1. **Launch the Application** – GUI will open.
2. **Select Camera** or **Load Video File**.
3. Click **Start Detection** to begin.
4. Detection results will be displayed with bounding boxes and alert panel.
5. Use **Train Model** to train the behavioral LSTM model with a custom dataset.

---

## 📊 System Workflow

**Input Source (Camera/Video) → YOLOv8 Object Detection → Weapon Classification → Pose Detection → LSTM Behavior Analysis → GUI Alert Overlay**

---

## 📸 Detection Legend

| Color     | Meaning               |
| --------- | --------------------- |
| 🟥 Red    | Real Weapon           |
| 🟨 Yellow | Potential Fake Weapon |
| 🟧 Orange | Suspicious Object     |


```
