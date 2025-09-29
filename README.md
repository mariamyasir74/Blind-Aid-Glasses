# 👓 Blind Aid Glasses – Banknote Recognition + Text Reader

This project is part of the Blind Aid Glasses system, designed to help visually impaired users interact with their environment.

It integrates:

💵 **Banknote Recognition** → Detects and classifies Egyptian banknotes in real time using a YOLOv8 detector (for background removal) and a ResNet50 classifier (for denomination recognition).

📖 **Text Reader** → Reads printed or handwritten text in Arabic and English using OCR (Tesseract) and outputs it as speech feedback.

The project is optimized for real-time use on laptops and embedded devices (e.g., Raspberry Pi)# 💵 Banknote Recognition System (YOLOv8 + ResNet50)

This project implements a **real-time Egyptian banknote recognition system** that combines:

- **YOLOv8** 🦾 → for **banknote detection** (localizing the note in the camera frame and removing background noise).  
- **ResNet50** 🧠 → for **banknote classification** (recognizing the denomination of the detected note).  

The system is designed for **real-time usage** with high accuracy and efficiency, suitable for deployment on laptops.

---

## 📂 Project Structure
