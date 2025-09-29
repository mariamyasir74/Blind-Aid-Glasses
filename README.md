# 👓 Blind Aid Glasses – Banknote Recognition + Text Reader

This project is part of the Blind Aid Glasses system, designed to help visually impaired users interact with their environment.

It integrates:

💵 **Banknote Recognition** → Detects and classifies Egyptian banknotes in real time using a **YOLOv8** detector (for background removal) and a **ResNet50** classifier (for denomination recognition).

📖 **Text Reader** → Reads printed or handwritten text in **Arabic** and **English** using OCR (Tesseract) and outputs it as speech feedback.

The project is optimized for real-time use on laptops and could simply be modified to be deployed on embedded devices (e.g., Raspberry Pi).

---

## ⚙️ Installation

1. **Clone the repo**
```bash
git clone https://github.com/mariamyasir74/Blind-Aid-Glasses.git
cd blind-aid-glasses
```
2. **Create a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```
3. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Analysis

- **Dataset Evaluation**
```bash
python dataset evaluation.py
```

<img width="1645" height="235" alt="dataset analysis" src="https://github.com/user-attachments/assets/9d45a8e6-0e62-43ce-801a-7caadae90138" />

<img width="640" height="480" alt="Class Distribution" src="https://github.com/user-attachments/assets/591778eb-11e8-4775-a8e9-5e63c9bcc8e8" />

<img width="1500" height="800" alt="Random Samples per Class" src="https://github.com/user-attachments/assets/7f28dfc6-1ce3-46c0-8663-16bedf8ed925" />

<img width="1000" height="800" alt="t-SNE Visualization" src="https://github.com/user-attachments/assets/847d496f-8990-479f-9143-b49e6d57d376" />

---

## 🏋️ Training
- **Train ResNet50 Classifier**
```bash
python train_banknote_resnet50.py
```
- **Train YOLOv8 Detector**
```bash
python train_banknote_yolov8.py
```

---

## 📈 Models Evaluation
- **ResNet50**
```bash
python evaluate resnet50 model.py
```

<img width="950" height="689" alt="ResNet50 test accuracy" src="https://github.com/user-attachments/assets/d746e6c9-8288-42a2-8354-4d398fde9dc1" />

<img width="1200" height="500" alt="ResNet50 loss and accuracy" src="https://github.com/user-attachments/assets/3058d5ea-d70c-4301-9f17-6cd22fabb732" />

<img width="838" height="505" alt="ResNet50 classification report" src="https://github.com/user-attachments/assets/13ba0f2d-0a35-4c74-90a6-a81d47a41eb3" />

<img width="800" height="600" alt="ResNet50 Confusion Matrix" src="https://github.com/user-attachments/assets/b9bdee0f-50ac-4ab5-ac21-7b30eb65b9ff" />

<img width="1000" height="800" alt="ResNet50 T-SNE Visualization" src="https://github.com/user-attachments/assets/ca5fdf85-57d2-46a5-8556-0fc3c6e62aba" />


- **YOLOv8**
```bash
python evaluate yolov8 model.py
```

<img width="588" height="207" alt="YOLOV8 metrics" src="https://github.com/user-attachments/assets/1d1abc20-acaa-4a1c-9668-1b06a740e69f" />

<img width="640" height="480" alt="YOLOV8 evaluation" src="https://github.com/user-attachments/assets/2801383b-07b5-4fcb-99a2-4846c9a646cb" />

---

## 🚀 Real-Time Pipelines
💵 **Banknote Recognition**
- **ResNet50 only**
```bash
python real_time_banknote_resnet50_recognition.py
```

- **YOLOv8 only**
```bash
python real_time_banknote_yolov8_detection.py
```

- **YOLOv8 + ResNet50 (Background Removal + Classification)**
```bash
python real_time_banknote_resnet50_yolov8_recognition.py
```

---

## 📖 Text Reader (Arabic & English)
```bash
python arabic and english text reader.py
```

---

## 📌 Links
- Dataset for YOLOV8: https://universe.roboflow.com/custom-yolov8-ihpb2/new-egyptian-currency/dataset/3
- Dataset for ResNet50:
