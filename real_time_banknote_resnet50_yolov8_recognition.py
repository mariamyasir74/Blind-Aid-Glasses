import numpy as np
import cv2
import pyttsx3
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import resnet50
from collections import deque
from PIL import Image
import time

CLASSES = ['10', '100', '20', '200', '5', '50']
yolo = YOLO(r"D:\Mariam Graduation project\Blind Aid Glasses\runs\train\banknote_yolov8n\weights\best.pt")
resnet_model = models.load_model('ResNet50_banknote.h5')

def speak_arabic(audio):
    engine = pyttsx3.init()
    engine.setProperty('rate', 200)
    engine.setProperty('voice',
                       r'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\MSTTS_V110_arSA_NaayfM')
    engine.say(audio)
    engine.runAndWait()

def classify_crop(crop_bgr):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(crop_rgb).resize((224, 224))
    arr = img_to_array(img).astype("float32")
    arr = resnet50.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    preds = resnet_model.predict(arr, verbose=0)
    probs = tf.nn.softmax(preds[0]).numpy()
    best_idx = np.argmax(probs)
    best_prob = float(probs[best_idx])
    label = CLASSES[best_idx]
    return label, best_prob

def draw_label_box(frame, box, text, conf, color=(0, 255, 0)):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{text} {conf:.2f}"
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - h - 8), (x1 + w + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                2, cv2.LINE_AA)

def run_video(source=0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video source", source)
    prev_time = 0
    recent_preds = deque(maxlen=5)
    last_spoken_label = None
    last_spoken_time = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed reading frame. Exiting.")
            break
        orig_h, orig_w = frame.shape[:2]
        results = yolo(frame, conf=0.35, iou=0.45, imgsz=640)
        res = results[0]
        try:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
        except:
            xyxy, confs = [], []
        for box, score in zip(xyxy, confs):
            if score < 0.35:
                continue
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w - 1, x2), min(orig_h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            pred_label, pred_prob = classify_crop(crop)
            if pred_prob > 0.8:
                recent_preds.append(pred_label)
            if len(recent_preds) > 0:
                smoothed_pred = max(set(recent_preds), key=recent_preds.count)
            else:
                smoothed_pred = pred_label
            draw_label_box(frame, box, smoothed_pred, pred_prob)
            now = time.time()
            if smoothed_pred != last_spoken_label or (now - last_spoken_time) > 5:
                speak_arabic(f"{smoothed_pred} جنيه")
                last_spoken_label = smoothed_pred
                last_spoken_time = now
        cur_time = time.time()
        fps = 1.0 / (cur_time - prev_time) if prev_time else 0.0
        prev_time = cur_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2)
        cv2.imshow("YOLOv8 + ResNet50 Banknote Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            cv2.imwrite("debug_frame.png", frame)
            print("Saved debug_frame.png")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_video(0)