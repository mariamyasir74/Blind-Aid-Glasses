import os
import cv2
import numpy as np
from PIL import Image
import imagehash
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import random

DATASET_PATH = "dataset_multiclass_unsplit"
# 1. Class Balance Check
class_counts = {cls: len(os.listdir(os.path.join(DATASET_PATH, cls)))
                for cls in os.listdir(DATASET_PATH)}
print("Class distribution:", class_counts)
plt.bar(class_counts.keys(), class_counts.values())
plt.title("Class Distribution")
plt.xticks(rotation=45)
plt.savefig('Class Distribution.png')
plt.show()
# 2. Blurry Image Detection

def is_blurry(image_path, threshold=50):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return True, 0
    score = cv2.Laplacian(img, cv2.CV_64F).var()
    return score < threshold, score

blurry_images = []
for cls in os.listdir(DATASET_PATH):
    for fname in os.listdir(os.path.join(DATASET_PATH, cls)):
        path = os.path.join(DATASET_PATH, cls, fname)
        blurry, score = is_blurry(path)
        if blurry:
            blurry_images.append((path, score))
print(f"Found {len(blurry_images)} blurry images")
# 3. Duplicate Image Detection
hashes = {}
duplicates = []
for cls in os.listdir(DATASET_PATH):
    for fname in os.listdir(os.path.join(DATASET_PATH, cls)):
        path = os.path.join(DATASET_PATH, cls, fname)
        try:
            h = str(imagehash.phash(Image.open(path)))
            if h in hashes:
                duplicates.append((path, hashes[h]))
            else:
                hashes[h] = path
        except:
            continue
print(f"Found {len(duplicates)} duplicate pairs")
# 4. Image Size Check
sizes = []
for cls in os.listdir(DATASET_PATH):
    for fname in os.listdir(os.path.join(DATASET_PATH, cls)):
        path = os.path.join(DATASET_PATH, cls, fname)
        try:
            img = Image.open(path)
            sizes.append(img.size)
        except:
            continue
size_counts = Counter(sizes)
print("Most common sizes:", size_counts.most_common(5))
# 5. Show Random Samples per Class

def show_samples(dataset_path, samples_per_class=5):
    classes = os.listdir(dataset_path)
    plt.figure(figsize=(15, 8))
    for i, cls in enumerate(classes):
        files = os.listdir(os.path.join(dataset_path, cls))
        chosen = random.sample(files, min(samples_per_class, len(files)))
        for j, fname in enumerate(chosen):
            path = os.path.join(dataset_path, cls, fname)
            img = Image.open(path).convert("RGB").resize((128, 128))
            plt.subplot(len(classes), samples_per_class, i*samples_per_class + j + 1)
            plt.imshow(img)
            plt.axis("off")
            if j == 0:
                plt.ylabel(cls, fontsize=12)
    plt.suptitle("Random Samples per Class", fontsize=16)
    plt.savefig('Random Samples per Class.png')
    plt.show()

show_samples(DATASET_PATH)
# 6. t-SNE / PCA Visualization

def extract_features(dataset_path, img_size=(64,64), max_images=500):
    X, y = [], []
    classes = os.listdir(dataset_path)
    for cls in classes:
        files = os.listdir(os.path.join(dataset_path, cls))
        random.shuffle(files)
        for fname in files[:max_images // len(classes)]:
            path = os.path.join(dataset_path, cls, fname)
            try:
                img = Image.open(path).convert("L").resize(img_size) # grayscale
                arr = np.array(img).flatten() / 255.0
                X.append(arr)
                y.append(cls)
            except:
                continue
    return np.array(X), np.array(y)

X, y = extract_features(DATASET_PATH, max_images=1000)
print("Feature matrix shape:", X.shape)
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=50).fit_transform(X_scaled)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_embedded = tsne.fit_transform(pca)
plt.figure(figsize=(10, 8))
for cls in np.unique(y):
    idx = y == cls
    plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=cls, alpha=0.6)
plt.legend()
plt.title("t-SNE Visualization of Banknote Dataset")
plt.savefig('t-SNE Visualization.png')
plt.show()