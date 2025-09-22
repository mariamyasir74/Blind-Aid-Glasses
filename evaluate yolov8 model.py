from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt

model = YOLO("runs/train/banknote_yolov8n/weights/best.pt")
metrics = model.val()
print(metrics)
results = pd.read_csv("runs/train/banknote_yolov8n/results.csv")
print(results.columns)
results[['epoch', 'train/box_loss', 'train/cls_loss', 'metrics/mAP50(B)']].plot(x='epoch')
plt.savefig("YOLOV8 evaluation.png")
plt.show()