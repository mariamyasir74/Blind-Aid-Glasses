import cv2
import albumentations as A
import os

transform = A.Compose([A.RandomBrightnessContrast(p=0.3), A.Rotate(limit=30, p=0.7), A.MotionBlur(p=0.2),
                       A.GaussNoise(p=0.2), A.Perspective(scale=(0.05, 0.15), p=0.4),
                       A.RandomResizedCrop(size=(224, 224), scale=(0.6, 1.0), p=0.5),
                       A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=0.5)], p=1.0)
input_folder = r"D:\Mariam Graduation project\20_new"
output_folder = r"D:\Mariam Graduation project\generated_images"
os.makedirs(output_folder, exist_ok=True)
for i, filename in enumerate(os.listdir(input_folder)):
    img = cv2.imread(os.path.join(input_folder, filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for j in range(17):
        augmented = transform(image=img)["image"]
        cv2.imwrite(os.path.join(output_folder, f"{i}_{j}.jpg"), cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))