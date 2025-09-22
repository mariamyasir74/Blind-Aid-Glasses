import cv2
from ultralytics import YOLO
import time

model = YOLO("runs/train/banknote_yolov8n/weights/best.pt")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    start_time = time.time()
    results = model(frame, imgsz=416, conf=0.5)
    annotated_frame = results[0].plot()
    fps = 1 / (time.time() - start_time)
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)
    cv2.imshow("Banknote Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()