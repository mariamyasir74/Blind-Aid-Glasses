"""
Real-time pipeline:
1) YOLOv8 detects banknote(s) in frame using best.pt
2) Crop each detected box
3) Classify the crop with ResNet50.h5
4) Draw results on the frame and show/save
"""
import time
import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

CLASS_NAMES = ['10', '100', '20', '200', '5', '50']
yolo = YOLO("runs/train/banknote_yolov8n/weights/best.pt")
resnet_model = tf.keras.models.load_model("ResNet50.h5")
# If your ResNet outputs logits, we'll apply softmax. We'll treat output as probabilities.

def classify_crop(crop_bgr):
    """
    crop_bgr: cropped image in BGR (OpenCV) format
    returns: (pred_label, pred_prob)
    """
    # Convert BGR -> RGB
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    # Resize to model input
    img = Image.fromarray(crop_rgb).resize((224, 224))
    arr = img_to_array(img).astype("float32")
    # Preprocess like ResNet50 (subtract mean, etc.)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)  # shape (1, H, W, 3)
    preds = resnet_model.predict(arr, verbose=0)
    if preds.ndim == 2 and preds.shape[1] > 1:
        probs = tf.nn.softmax(preds[0]).numpy()
    else:
        # binary or single logit -> convert to prob
        try:
            probs = tf.nn.sigmoid(preds[0]).numpy()
            # If single output, convert to two-class-like output
            if probs.size == 1:
                # make two elements [1-prob, prob]
                probs = np.array([1.0 - probs[0], probs[0]])
        except:
            probs = np.array(preds[0])
    # Get best class
    best_idx = np.argmax(probs)
    best_prob = float(probs[best_idx])
    label = CLASS_NAMES[best_idx] if best_idx < len(CLASS_NAMES) else str(best_idx)
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
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed reading frame. Exiting.")
            break
        # Resize frame for display and possibly to speed up
        # Keep a copy for cropping using original resolution for better results
        orig_h, orig_w = frame.shape[:2]
        # YOLO inference. Using yolo(frame) returns a list of Results; take first
        # We pass imgsz to control inference resolution
        results = yolo(frame, conf=0.35, iou=0.45, imgsz=640)  # list-like
        # results could be a Results object or list; handle generically
        # Each result corresponds to the frame (we have a single frame)
        res = results[0]
        # Boxes: res.boxes.xyxy  (Tensor Nx4), res.boxes.conf (N), res.boxes.cls (N)
        boxes = []
        scores = []
        classes = []
        try:
            xyxy = res.boxes.xyxy.cpu().numpy()  # (N,4)
            confs = res.boxes.conf.cpu().numpy()
            cls_ids = res.boxes.cls.cpu().numpy().astype(int)
            for b, c, cid in zip(xyxy, confs, cls_ids):
                boxes.append(b)       # [x1,y1,x2,y2]
                scores.append(float(c))
                classes.append(int(cid))  # if you trained YOLO on "banknote" class, class id probably 0
        except Exception as e:
            # If no detections or different structure, skip
            boxes = []
            scores = []
            classes = []
        # For each detection, crop and classify
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(int, box)
            # Safety clamp
            x1 = max(0, min(x1, orig_w - 1))
            x2 = max(0, min(x2, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            y2 = max(0, min(y2, orig_h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]  # BGR
            # Optional: enlarge crop slightly to include context
            pad = 8
            x1p = max(0, x1 - pad); y1p = max(0, y1 - pad)
            x2p = min(orig_w - 1, x2 + pad); y2p = min(orig_h - 1, y2 + pad)
            crop = frame[y1p:y2p, x1p:x2p]
            # Classify cropped note
            try:
                pred_label, pred_prob = classify_crop(crop)
            except Exception as e:
                pred_label, pred_prob = "err", 0.0

            draw_label_box(frame, (x1, y1, x2, y2), pred_label, pred_prob)
        # FPS
        cur_time = time.time()
        fps = 1.0 / (cur_time - prev_time) if prev_time else 0.0
        prev_time = cur_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2)
        cv2.imshow("YOLOv8 -> ResNet50 Banknote Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        # press 's' to save a screenshot for debugging
        if key == ord("s"):
            cv2.imwrite("debug_frame.png", frame)
            print("Saved debug_frame.png")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_video(0)