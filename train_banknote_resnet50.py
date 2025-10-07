import os
import pickle
from PIL import ImageFile
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import resnet50, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, regularizers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

ImageFile.LOAD_TRUNCATED_IMAGES = True
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
train_valid_dir = os.path.join(THIS_FOLDER, "dataset_multiclass/train_valid")
test_dir = os.path.join(THIS_FOLDER, "dataset_multiclass/test")
train_datagen = ImageDataGenerator(preprocessing_function=resnet50.preprocess_input, validation_split=0.2)
test_datagen = ImageDataGenerator(preprocessing_function=resnet50.preprocess_input)
train_images = train_datagen.flow_from_directory(directory=train_valid_dir, target_size=(224, 224), color_mode='rgb',
                                                 batch_size=32, class_mode='categorical', shuffle=True, seed=42,
                                                 subset='training')
val_images = train_datagen.flow_from_directory(directory=train_valid_dir, target_size=(224, 224), color_mode='rgb',
                                               batch_size=32, class_mode='categorical', shuffle=True, seed=42,
                                               subset='validation')
test_images = test_datagen.flow_from_directory(directory=test_dir, target_size=(224, 224), color_mode='rgb',
                                               batch_size=32, class_mode='categorical', shuffle=False)
print('Classes indices:\n', train_images.class_indices, '\n', val_images.class_indices, '\n', test_images.class_indices)
num_classes = train_images.num_classes
print("Number of classes:", num_classes)
pretrained_model = ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet', pooling=None)
inputs = pretrained_model.input
x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pretrained_model.output)
x = layers.BatchNormalization()(x)
x = layers.GlobalAveragePooling2D()(x)
se = layers.Dense(512, activation='relu')(x)
se = layers.Dense(512, activation='sigmoid')(se)
x = layers.Multiply()([x, se])
x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)
model = Model(inputs, outputs)
print(model.summary())
for layer in pretrained_model.layers:
    layer.trainable = False
model.compile(optimizer=Adam(1e-3),loss='categorical_crossentropy',metrics=['accuracy'])
callbacks = [EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
             ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)]
history = model.fit(train_images, validation_data=val_images, epochs=10, callbacks=callbacks)
with open("history.pkl", "wb") as f:
    pickle.dump(history.history, f)
for layer in pretrained_model.layers:
    if layer.name.startswith("conv5_block") or layer.name.startswith("conv4_block6"):
        layer.trainable = True
    else:
        layer.trainable = False
model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
fine_tuned_history = model.fit(train_images, validation_data=val_images, epochs=20, callbacks=callbacks)
with open("fine_tuned_history.pkl", "wb") as f:
    pickle.dump(fine_tuned_history.history, f)
results = model.evaluate(test_images, verbose=0)
print("Test Loss:", results[0])
print("Test Accuracy:", np.round(results[1] * 100, 2), "%")
model.save(os.path.join(THIS_FOLDER, "ResNet50_banknote.h5"))
model.save_weights(os.path.join(THIS_FOLDER, "ResNet50_banknote.weights.h5"))
print("model, weights and histories saved")
total_loss = history.history['loss'] + fine_tuned_history.history['loss']
total_val_loss = history.history['val_loss'] + fine_tuned_history.history['val_loss']
total_acc = history.history['accuracy'] + fine_tuned_history.history['accuracy']
total_val_acc = history.history['val_accuracy'] + fine_tuned_history.history['val_accuracy']
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(total_loss, label='Train Loss')
plt.plot(total_val_loss, label='Test Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.axvline(x=len(history.history['loss']) - 1, color='gray', linestyle='--', label='Fine-tuning start')
plt.legend()
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(total_acc, label='Train Accuracy')
plt.plot(total_val_acc, label='Test Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.axvline(x=len(history.history['loss']) - 1, color='gray', linestyle='--', label='Fine-tuning start')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('ResNet50 loss and accuracy.png')
plt.show()
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("ResNet50_banknote.tflite", "wb").write(tflite_model)