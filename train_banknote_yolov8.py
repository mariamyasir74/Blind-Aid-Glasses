from ultralytics import YOLO

def main():
    # 1. Make sure you have a data.yaml like this:
    #
    # train: path/to/train/images
    # val:   path/to/val/images
    # nc:    6
    # names: ['5EGP','10EGP','20EGP','50EGP','100EGP','200EGP']
    # Adjust paths and class names to match your folders.
    data_yaml = "dataset/data.yaml"
    # 2. Choose your model (nano / small / medium)
    #    Here we pick nano for speed. You can also try 'yolov8s.pt' or 'yolov8m.pt'.
    model = YOLO("yolov8n.pt")
    # 3. Train!
    results = model.train(
        data=data_yaml,
        imgsz=416,         # input size (reduce if you need more speed)
        batch=16,          # batch size (lower if you run out of memory)
        epochs=30,         # number of epochs
        lr0=0.001,         # initial learning rate
        optimizer="Adam",  # optimizer: Adam or SGD
        patience=5,       # early stopping patience
        device="cpu",          # change to 'cpu' or index of GPU
        augment=True,      # enable built-in augmentations (mosaic, mixup, HSV, etc.)
        name="banknote_yolov8n",  # runs/<name> folder
        project="runs/train"
    )
    # 4. After training, best weights will be at:
    #    runs/train/banknote_yolov8n/weights/best.pt
    # 5. (Optional) Export the model for deployment:
    model.export(format="onnx")   # creates best.onnx
    model.export(format="tflite") # creates best.tflite
if __name__ == "__main__":
    main()