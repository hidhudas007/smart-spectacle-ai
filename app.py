from flask import Flask, Response
from ultralytics import YOLO
import cv2
import face_recognition
import os
import threading
import numpy as np
import pytesseract
import time
import subprocess
import mediapipe as mp

app = Flask(__name__)

# Camera config
camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Load YOLOv8
model = YOLO("yolov8n.pt")

# Text-to-speech
def speak(text):
    subprocess.run(["espeak", text])

# Globals
last_gesture = ""
last_spoken_time = {}
SPEAK_INTERVAL = 5  # seconds
YOLO_CONFIDENCE_THRESHOLD = 0.6  # Speak if confidence > 60%

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Gesture classification
def classify_gesture(landmarks):
    thumb_tip = landmarks[4].y
    index_tip = landmarks[8].y
    middle_tip = landmarks[12].y
    ring_tip = landmarks[16].y
    pinky_tip = landmarks[20].y
    if all(tip < landmarks[0].y for tip in [index_tip, middle_tip, ring_tip, pinky_tip]) and thumb_tip > landmarks[2].y:
        return "Palm"
    elif thumb_tip < index_tip and all(tip > landmarks[0].y for tip in [index_tip, middle_tip, ring_tip, pinky_tip]):
        return "Thumbs Up"
    return None

# Load known faces
known_face_encodings = []
known_face_names = []
face_dir = "faces"
for filename in os.listdir(face_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(face_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])

def gen_frames():
    global last_gesture
    while True:
        ret, frame = camera.read()
        if not ret:
            continue

        current_time = time.time()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face Recognition
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                idx = matches.index(True)
                name = known_face_names[idx]
                if name not in last_spoken_time or current_time - last_spoken_time[name] > SPEAK_INTERVAL:
                    threading.Thread(target=speak, args=(f"{name} detected",), daemon=True).start()
                    last_spoken_time[name] = current_time
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # YOLO Object Detection
        results = model(frame, verbose=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                label = model.names[class_id]
                if confidence > YOLO_CONFIDENCE_THRESHOLD:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    # Speak object name
                    if label not in last_spoken_time or current_time - last_spoken_time[label] > SPEAK_INTERVAL:
                        threading.Thread(target=speak, args=(f"{label} detected",), daemon=True).start()
                        last_spoken_time[label] = current_time

        # Gesture Recognition
        result = hands.process(rgb_frame)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                gesture = classify_gesture(hand_landmarks.landmark)
                if gesture:
                    if gesture != last_gesture and (gesture not in last_spoken_time or current_time - last_spoken_time[gesture] > SPEAK_INTERVAL):
                        threading.Thread(target=speak, args=(gesture,), daemon=True).start()
                        last_spoken_time[gesture] = current_time
                        last_gesture = gesture
                    cv2.putText(frame, f'Gesture: {gesture}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # OCR
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        if text.strip():
            line = text.strip().split('\n')[0]
            if "ocr" not in last_spoken_time or current_time - last_spoken_time["ocr"] > SPEAK_INTERVAL:
                threading.Thread(target=speak, args=(f"Text detected: {line}",), daemon=True).start()
                last_spoken_time["ocr"] = current_time
            cv2.putText(frame, line[:40], (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Stream frame
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return '''
        <html><head><title>Detection App</title></head>
        <body>
            <h1>YOLO + Face + OCR + Gesture</h1>
            <img src="/video" width="640" height="480">
        </body></html>
    '''

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return "Server running", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
