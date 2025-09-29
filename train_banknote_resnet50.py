import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.optimizers import Adam
from PIL import ImageFile
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
train_valid_dir = os.path.join(THIS_FOLDER, "dataset_multiclass/train_valid")
test_dir = os.path.join(THIS_FOLDER, "dataset_multiclass/test")
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input, validation_split=0.2)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input)
train_images = train_datagen.flow_from_directory(directory=train_valid_dir, target_size=(224, 224), color_mode='rgb',
                                                 batch_size=32, class_mode='categorical', shuffle=True, seed=42,
                                                 subset='training')
val_images = train_datagen.flow_from_directory(directory=train_valid_dir, target_size=(224, 224), color_mode='rgb',
                                               batch_size=32, class_mode='categorical', shuffle=True, seed=42,
                                               subset='validation')
test_images = test_datagen.flow_from_directory(directory=test_dir, target_size=(224, 224), color_mode='rgb',
                                               batch_size=32, class_mode='categorical', shuffle=False)
num_classes = train_images.num_classes
print("Number of classes:", num_classes)
pretrained_model = tf.keras.applications.ResNet50(input_shape=(224, 224, 3), include_top=False,
                                                  weights='imagenet', pooling='avg')
pretrained_model.trainable = False
inputs = pretrained_model.input
x = layers.BatchNormalization()(pretrained_model.output)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)
print(model.summary())
model.compile(optimizer=Adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
history = model.fit(train_images, validation_data=val_images, epochs=20, callbacks=callbacks)
model.save(os.path.join(THIS_FOLDER, "ResNet50.h5"))
print("model saved")
results = model.evaluate(test_images, verbose=0)
print("Test Loss:", results[0])
print("Test Accuracy:", np.round(results[1] * 100, 2), "%")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('ResNet50 loss and accuracy.png')
plt.show()