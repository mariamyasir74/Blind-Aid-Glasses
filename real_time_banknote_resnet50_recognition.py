import cv2
import numpy as np
import pyttsx3
import threading
import tensorflow as tf
from tensorflow.keras import models

def speak_arabic(audio):
    engine = pyttsx3.init()
    engine.setProperty('rate', 200)
    engine.setProperty('voice',
                       'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\MSTTS_V110_arSA_NaayfM')
    engine.say(audio)
    engine.runAndWait()

def async_speech(audio):
    speech_thread = threading.Thread(target=speak_arabic, args=(audio,))
    speech_thread.start()

def draw_label(img, text, pos, bg_color):
    margin = 2
    txt_size = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=cv2.FILLED)
    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin
    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness=cv2.FILLED)
    cv2.putText(img, text, pos, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=2,
                lineType=cv2.LINE_AA)

model = models.load_model('ResNet50.h5')
Labels = ["Ten", "One Hundred", "Twenty", "Two Hundred", "Five", "Fifty"]
Labels_ar = ['عشرة جنيه', 'مئة جنيه', 'عشرون جنيه', 'مئتاا جنيه', 'خمسة جنيه', 'خمسون جنيه']
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open video device")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break
    try:
        image = cv2.resize(frame, (224, 224))
        image = np.expand_dims(image, axis=0)
        image = tf.keras.applications.resnet50.preprocess_input(image)
        prediction = model.predict(image, verbose=0)[0]
        class_index = np.argmax(prediction)
        confidence = prediction[class_index]
        if Labels[class_index] != "none" and confidence > 0.50:
            draw_label(frame, f'Label: {Labels[class_index]} ({confidence * 100:.1f}%)', (20, 20),
                       (255, 0, 0))
            async_speech((Labels_ar[class_index]))
        cv2.imshow('Currency Recognition', frame)
    except Exception as e:
        print(f"Error during prediction: {e}")
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()