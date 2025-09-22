import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
train_valid_dir = os.path.join(THIS_FOLDER, "dataset_multiclass/train_valid")
test_dir = os.path.join(THIS_FOLDER, "dataset_multiclass/test")
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input)
test_images = test_datagen.flow_from_directory(directory=test_dir, target_size=(224, 224), color_mode='rgb',
                                               batch_size=32, class_mode='categorical', shuffle=False)
model = load_model(os.path.join(THIS_FOLDER, "ResNet50.h5"))
y_true = test_images.classes
y_pred_probs = model.predict(test_images, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
class_labels = list(test_images.class_indices.keys())
print(classification_report(y_true, y_pred, target_names=class_labels))
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig('ResNet50 Confusion Matrix.png')
plt.show()

def visualize_tsne(train_valid_dir):
    feature_extractor = tf.keras.applications.resnet50.ResNet50(include_top=False, pooling='avg', weights='imagenet')
    features, labels = [], []
    for class_label in os.listdir(train_valid_dir):
        class_dir = os.path.join(train_valid_dir, class_label)
        for image_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, image_name)
            img = cv2.imread(img_path)
            if img is None: continue
            img = cv2.resize(img, (224, 224))
            img = np.expand_dims(img, axis=0)
            img = tf.keras.applications.resnet50.preprocess_input(img)
            feature = feature_extractor.predict(img)
            features.append(feature.flatten())
            labels.append(class_label)
    features = np.array(features)
    features = StandardScaler().fit_transform(features)
    reduced_features = TSNE(n_components=2, random_state=42).fit_transform(features)
    plt.figure(figsize=(10, 8))
    for cls in class_labels:
        indices = [i for i, label in enumerate(labels) if label == cls]
        plt.scatter(reduced_features[indices, 0], reduced_features[indices, 1], label=cls, alpha=0.6)
    plt.legend()
    plt.title("ResNet50 t-SNE Visualization of features.png")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.savefig('ResNet50 T-SNE Visualization.png')
    plt.show()

visualize_tsne(train_valid_dir)