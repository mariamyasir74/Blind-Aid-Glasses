import cv2
import pytesseract
from pytesseract import Output
import pyttsx3

pytesseract.pytesseract.tesseract_cmd = 'D:/Tesseract-OCR/tesseract.exe'

def speak(audio):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    engine.say(audio)
    engine.runAndWait()

def speak_arabic(audio):
    engine = pyttsx3.init()
    engine.setProperty('rate', 200)
    engine.setProperty('voice',
                       r'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\MSTTS_V110_arSA_NaayfM')
    engine.say(audio)
    engine.runAndWait()

def correct_arabic_text(text):
    corrections = {
        'للهننسة': 'للهندسة',
        'التكنملم': 'التكنولوجيا',
        'بلمنصورة': 'بالمنصورة',
        'بمصورة': 'بالمنصورة',
        'بالمتصورة': 'بالمنصورة',
        'سريم': 'مريم',
        'يسر': 'ياسر',
        'يلسر': 'ياسر',
        'سليسان': 'سليمان',
        'الفرياو': 'الغرباوي',
        'الغرياو': 'الغرباوي',
        'الغرباو': 'الغرباوي',
        'الفرف': 'الفرقة',
        'بعة': 'الرابعة',
        'الداالرابعة': 'الرابعة',
        'الراالرابعة': 'الرابعة',
        'االرابعة': 'الرابعة',
        'النهانية': 'النهائية',
        'التعهية': 'النهائية',
        'النهفية': 'النهائية',
        'التهفية': 'النهائية',
        'النهلنية': 'النهائية',
        'التهائية': 'النهائية',
        'التهانية': 'النهائية',
        'منسة': 'هندسة',
        'فسة': 'هندسة',
        'فنسة': 'هندسة',
        'فننسة': 'هندسة',
        'هننسة': 'هندسة',
        'الحسبت': 'الحاسبات',
        'الحلسبت': 'الحاسبات',
        'الحسيت': 'الحاسبات',
        'العسيت': 'الحاسبات',
        'اللية': 'الآلية',
        'رقر': 'رقم',
        'رفر': 'رقم',
        'رلم': 'رقم',
        'رفم': 'رقم',
        'الطعب': 'الطالب',
        'تصري': 'تصريح',
        'حاستخدا': 'استخدام',
        'يشري': 'بشري',
        'ماسانسير': 'اسانسير',
    }
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    return text

cap = cv2.VideoCapture(0)
print("Real-time Multilingual OCR started. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    config = '--psm 6 --oem 3 -l ara+eng'
    data = pytesseract.image_to_data(frame, config=config, output_type=Output.DICT)
    full_arabic = ''
    full_english = ''
    for i in range(len(data['text'])):
        text = data['text'][i]
        conf = int(data['conf'][i])
        if conf > 65 and text.strip():
            is_arabic = any('\u0600' <= c <= '\u06FF' for c in text)
            if is_arabic:
                text = correct_arabic_text(text)
                full_arabic += text + ' '
            else:
                full_english += text + ' '
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    combined_text = f"AR: {full_arabic.strip()}\nEN: {full_english.strip()}"
    print(combined_text)
    if full_arabic.strip():
        speak_arabic(full_arabic)
    if full_english.strip():
        speak(full_english)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()