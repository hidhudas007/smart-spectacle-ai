# Smart Spectacle AI 👓🎯

A wearable AI-powered prototype to assist visually impaired individuals by detecting objects, faces, gestures, and printed text — and speaking them aloud in real-time using Raspberry Pi.

---

## 🚀 Inspiration
We were inspired by the daily struggles of visually impaired people and wanted to create a wearable device that could assist them using computer vision and voice.

---

## 🔍 What it does
- Detects objects (YOLOv8)
- Recognizes faces (dlib)
- Detects hand gestures (MediaPipe)
- Reads text from the environment (OCR)
- Converts everything into real-time speech output

---

## 🛠️ How we built it
Built using:
- Python
- OpenCV
- Ultralytics YOLOv8
- Dlib + face_recognition
- MediaPipe
- Pytesseract (OCR)
- eSpeak (TTS)
- Flask (Web stream)
- Raspberry Pi 4 (Raspbian OS)

---

## ⚠️ Challenges
- Real-time performance on low-power device
- Thread-safe audio feedback
- Gesture detection in dynamic lighting
- Smooth integration of all modules

---

## 🏆 Accomplishments
- Fully offline working prototype
- Detects and speaks out multiple real-world inputs
- Simple interface via browser

---

## 📚 What we learned
- Edge AI optimization
- Real-time video processing
- Threading in Python
- Multi-module integration on embedded platforms

---

## 🚧 What’s next?
- GPS integration
- Multilingual support
- Voice command interaction
- Cloud sync for face/object data

---

## 🔗 Try it out
- [Demo Video](https://youtu.be/your-demo-link)
- [Local Flask App](http://<your-pi-ip>:5000)

---

## 👨‍💻 Contributors
- Hidhu Das P P
- Mohammed Aadhil S
- Dileep K M
- Prajeesh K
- Guided by Ms. Reshna S, Associate Professor, Dept. of ECE

---

## License
MIT
