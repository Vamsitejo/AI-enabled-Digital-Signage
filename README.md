# AI Enabled Digital Signage

An AI-powered digital signage system that performs **real-time face detection and age & gender classification** to display **targeted advertisements** based on the detected audience profile.

This project is designed for **kiosks, retail stores, malls, and public displays**, enabling smarter and more personalized advertising using computer vision.

---

## 📌 Project Overview

The system captures live video input, detects faces, predicts **age range and gender**, and displays suitable **image/GIF advertisements** mapped to predefined audience categories.

Example targeting logic:

- **Age group:** 25–32
- **Gender:** Male / Female
- **Ad type:** Image or GIF

---

## 📁 Project Structure

```
AI-enabled-Digital-Signage/
│
├── ads/                          # Advertisement assets
│   └── 25-32/
│       ├── Male/                # Ads for males (images/GIFs)
│       └── Female/              # Ads for females (images/GIFs)
│
├── outputs/                      # Stored output frames / logs / screenshots
│
├── age_deploy.prototxt           # Age model architecture
├── age_net.caffemodel            # Pre-trained age classification model
│
├── gender_deploy.prototxt        # Gender model architecture
├── gender_net.caffemodel         # Pre-trained gender classification model
│
├── opencv_face_detector.pbtxt    # Face detector config
├── opencv_face_detector_uint8.pb # Face detector weights
│
├── detect_alone.py               # Standalone age & gender detection script
├── digital_signage_kiosk.py      # Main digital signage application
│
├── homepage.png                  # Kiosk home screen image
└── README.md                     # Project documentation
```

---

## 🧠 Models Used

- **Face Detection:** OpenCV DNN face detector
- **Age Classification:** Caffe-based CNN model
- **Gender Classification:** Caffe-based CNN model

These models are lightweight and suitable for **real-time inference** on edge devices.

---

## ⚙️ Requirements

### Python Libraries Used

The following libraries are used across the project:

- `opencv-python`
- `numpy`
- `PyQt5`
- `ffpyplayer`
- `collections` (inbuilt)
- `glob` (inbuilt)
- `random` (inbuilt)
- `time`, `os`, `sys` (inbuilt)

Install required third-party dependencies using:

```bash
pip install opencv-python numpy
```

---

## ▶️ How to Run

### 1️⃣ Standalone Age & Gender Detection

```bash
python detect_alone.py
```

This mode is useful when the user wants to **only visualize age and gender predictions** in a simple window without running the full digital signage system.

Features:
- Live camera feed
- Face detection
- Age prediction
- Gender prediction
- Bounding boxes with labels

---



### 2️⃣ Run Digital Signage Kiosk

```bash
python digital_signage_kiosk.py
```

This is the **full digital signage mode**.

Features:
- Real-time face detection
- Age & gender classification
- Dynamic ad selection
- Image/GIF ad playback using PyQt5
- Media rendering using FFpyPlayer

The displayed advertisement is selected automatically from the `ads/` directory

---

## 🖼️ Advertisement Logic

Ads are selected based on:

- **Predicted age group**
- **Predicted gender**

Example directory mapping:

```
ads/
└── 25-32/
    ├── Male/
    │   ├── ad1.jpg
    │   └── ad2.gif
    └── Female/
        ├── ad1.jpg
        └── ad2.gif
```

You can easily extend this structure to support:
- More age groups
- Different locations
- Product-based campaigns

---

## 📤 Outputs

The `outputs/` folder contains:
- Processed frames
- Logs or screenshots (if enabled)
- Debug or analytics outputs

---

## 🚀 Applications

- Retail stores
- Shopping malls
- Digital kiosks
- Smart advertising boards
- Audience analytics systems

---

## 🔮 Future Enhancements

- Emotion-based ad targeting
- Audience count & dwell time
- Cloud-based analytics dashboard
- Multi-camera support
- Database logging

---

## 📜 License

This project is intended for **educational and research purposes**.

---

## 👤 Author

**Vamsi Tejo**

If you like this project, feel free to ⭐ the repository!

