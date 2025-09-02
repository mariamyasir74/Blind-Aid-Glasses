import cv2
from ultralytics import YOLO
import time

# 1. Load your trained model
model = YOLO("runs/train/banknote_yolov8n/weights/best.pt")  # update path if needed

# 2. Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Optional: set webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    start_time = time.time()

    # 3. Run YOLO inference on the frame
    results = model(frame, imgsz=416, conf=0.5)  # conf = confidence threshold

    # 4. Annotate frame with detection results
    annotated_frame = results[0].plot()

    # 5. Calculate FPS
    fps = 1 / (time.time() - start_time)
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 6. Display
    cv2.imshow("Banknote Detection", annotated_frame)

    # 7. Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
