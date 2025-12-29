# =====================================
# AGE AND GENDER BASED DIGITAL SIGNAGE
# =====================================


import sys
import os
import time
import cv2
import numpy as np
import glob
import random
from collections import Counter, deque

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal, QThread
from ffpyplayer.player import MediaPlayer

# ---------- CONFIG ----------
FACE_PROTO = "opencv_face_detector.pbtxt"
FACE_MODEL = "opencv_face_detector_uint8.pb"
AGE_PROTO = "age_deploy.prototxt"
AGE_MODEL = "age_net.caffemodel"
GENDER_PROTO = "gender_deploy.prototxt"
GENDER_MODEL = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']


WEBCAM_INDEX = 0
DETECTION_SECONDS = 2.0
ADS_DIR = "ads"
HOMEPAGE_PATH = "homepage.png"
PADDING = 20
WORKER_SLEEP = 0.02
MIN_FACE_BOX_HEIGHT = 80
POSTER_DISPLAY_TIME = 10  # seconds
FACE_CONF_THRESHOLD = 0.3
CONFIDENCE_THRESHOLD = 0.5  # for age/gender predictions
MAX_HISTORY = 5
WORKER_SLEEP = 0.03  # 30 ms

# Temporal smoothing cache
prediction_history = {}

# ---------------- Helper Functions ----------------

def enhance_lighting(face):
    """Apply CLAHE for lighting normalization."""
    face_ycrcb = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(face_ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y = clahe.apply(y)
    face_clahe = cv2.merge((y, cr, cb))
    return cv2.cvtColor(face_clahe, cv2.COLOR_YCrCb2BGR)


def detect_faces(net, frame, conf_threshold=0.6):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            boxes.append((max(0, x1), max(0, y1),
                          min(w - 1, x2), min(h - 1, y2)))
    return boxes


def predict_age_gender(face, age_net, gender_net):
    """Predict age and gender using ensemble (original + flipped)."""
    face = enhance_lighting(face)

    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                 MODEL_MEAN_VALUES, swapRB=False)
    flipped_face = cv2.flip(face, 1)
    blob_flipped = cv2.dnn.blobFromImage(flipped_face, 1.0, (227, 227),
                                         MODEL_MEAN_VALUES, swapRB=False)

    # Gender prediction
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender_net.setInput(blob_flipped)
    gender_preds_flip = gender_net.forward()
    gender_avg = (gender_preds[0] + gender_preds_flip[0]) / 2
    gender_id = int(np.argmax(gender_avg))
    gender_label = GENDER_LIST[gender_id]
    gender_conf = gender_avg[gender_id]

    # Age prediction
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age_net.setInput(blob_flipped)
    age_preds_flip = age_net.forward()
    age_avg = (age_preds[0] + age_preds_flip[0]) / 2
    age_id = int(np.argmax(age_avg))
    age_label = AGE_LIST[age_id]
    age_conf = age_avg[age_id]

    return gender_label, gender_conf, age_label, age_conf


# ---------------- Worker Thread ----------------

class DetectionWorker(QThread):
    step = pyqtSignal(object, object, bool)  # frame, detections, person_present

    def __init__(self, webcam_index, face_net, age_net, gender_net):
        super().__init__()
        self.webcam_index = webcam_index
        self.face_net = face_net
        self.age_net = age_net
        self.gender_net = gender_net
        self.running = False
        self.cap = None
        self.frame_counter = 0
        self.recent_faces = deque(maxlen=10)
        self.face_present_time = 0

    def run(self):
        self.cap = cv2.VideoCapture(self.webcam_index)
        if not self.cap.isOpened():
            print("❌ Failed to open camera")
            return

        print("✅ DetectionWorker started...")
        self.running = True

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            debug_frame = frame.copy()
            boxes = detect_faces(self.face_net, frame, FACE_CONF_THRESHOLD)
            detections = []

            for (x1, y1, x2, y2) in boxes:
                face_height = y2 - y1
                print(face_height)
                if face_height < MIN_FACE_BOX_HEIGHT:
                    continue

                face = frame[max(0, y1 - PADDING):min(y2 + PADDING, frame.shape[0] - 1),
                             max(0, x1 - PADDING):min(x2 + PADDING, frame.shape[1] - 1)]

                if face.size == 0:
                    continue

                gender_label, gender_conf, age_label, age_conf = predict_age_gender(
                    face, self.age_net, self.gender_net)

                # Confidence filtering
                if gender_conf < CONFIDENCE_THRESHOLD or age_conf < CONFIDENCE_THRESHOLD:
                    continue

                # Temporal smoothing
                face_key = (x1 // 10, y1 // 10)
                if face_key not in prediction_history:
                    prediction_history[face_key] = {
                        'gender': deque(maxlen=MAX_HISTORY),
                        'age': deque(maxlen=MAX_HISTORY)
                    }

                prediction_history[face_key]['gender'].append(gender_label)
                prediction_history[face_key]['age'].append(age_label)

                gender_label = Counter(prediction_history[face_key]['gender']).most_common(1)[0][0]
                age_label = Counter(prediction_history[face_key]['age']).most_common(1)[0][0]

                gender_id = GENDER_LIST.index(gender_label)
                age_id = AGE_LIST.index(age_label)
                detections.append((gender_id, age_id))

                label = f"{gender_label}, {age_label}"
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(debug_frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            current_time = time.time()
            if detections:
                self.recent_faces.append(current_time)
                self.face_present_time = current_time

            person_present = (current_time - self.face_present_time) < 1.0

            try:
                self.step.emit(debug_frame, detections, person_present)
            except Exception:
                pass

            time.sleep(WORKER_SLEEP)

        if self.cap and self.cap.isOpened():
            self.cap.release()
        print("🛑 DetectionWorker stopped.")

    def stop(self):
        self.running = False
        self.wait(2000)


# ---------- Main Window ----------
class SmartAdApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Ad Player - Fullscreen Kiosk")
        self.showFullScreen()
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)

        # Load homepage
        if os.path.exists(HOMEPAGE_PATH):
            self.homepage = cv2.imread(HOMEPAGE_PATH)
        else:
            self.homepage = np.zeros((540, 960, 3), np.uint8)

        # Load models
        self.face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
        self.age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
        self.gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)

        # Central layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Video area
        self.video_label = QtWidgets.QLabel()
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.video_label, stretch=10)

        # Scrolling text banner
        self.scroll_label = QtWidgets.QLabel("")
        self.scroll_label.setAlignment(QtCore.Qt.AlignVCenter)
        self.scroll_label.setStyleSheet("""
            background-color: rgba(0, 0, 0, 180);
            color: white;
            font-size: 28px;
            padding: 10px;
            font-weight: bold;
        """)
        layout.addWidget(self.scroll_label, stretch=1)

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.BlankCursor)

        # Detection worker
        self.worker = DetectionWorker(WEBCAM_INDEX, self.face_net, self.age_net, self.gender_net)
        self.worker.step.connect(self.on_worker_step)
        self.worker.start()

        # State vars
        self.person_present = False
        self.last_seen = time.time()
        self.playing_ad = False
        self.detect_start_time = None
        self.collected_detections = []
        self.ad_playlist = []
        self.ad_index = 0
        self.current_age_label = None
        self.scroll_text = ""
        self.scroll_pos = 0

        # Scroll ticker
        self.scroll_timer = QtCore.QTimer()
        self.scroll_timer.timeout.connect(self.update_scroll_text)
        self.scroll_timer.start(100)

        self.show_homepage()

    # === UI and Scroll Logic ===
    def update_scroll_text(self):
        if not self.scroll_text:
            return
        display_text = self.scroll_text[self.scroll_pos:] + " " + self.scroll_text[:self.scroll_pos]
        self.scroll_label.setText(display_text)
        self.scroll_pos = (self.scroll_pos + 1) % len(self.scroll_text)

    def update_ui_frame(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        qimg = QtGui.QImage(frame_rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pix.scaled(
            self.video_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def show_homepage(self):
        if self.playing_ad:
            return
        rgb = cv2.cvtColor(self.homepage, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0],
                            QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.video_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.video_label.setPixmap(pix)
        self.scroll_text = " Waiting for person detection... "
        self.scroll_pos = 0

    # === Detection Logic ===
    def on_worker_step(self, debug_frame, detections, person_present):
        if person_present:
            self.last_seen = time.time()
            self.person_present = True
        else:
            self.person_present = False

        if self.playing_ad:
            if (time.time() - self.last_seen) > 1.5:
                self.stop_posters()
            return

        if detections:
            if self.detect_start_time is None:
                self.detect_start_time = time.time()
                self.collected_detections = []
            self.collected_detections.extend(detections)
            elapsed = time.time() - self.detect_start_time
            if elapsed >= DETECTION_SECONDS:
                genders = [g for (g, a) in self.collected_detections]
                ages = [a for (g, a) in self.collected_detections]
                if len(ages) > 0:
                    age_id = Counter(ages).most_common(1)[0][0]
                    gender_id = Counter(genders).most_common(1)[0][0]
                    self.start_ad(age_id, gender_id)
                self.detect_start_time = None
                self.collected_detections = []
        else:
            if self.detect_start_time and (time.time() - self.last_seen > 1.0):
                self.detect_start_time = None
                self.collected_detections = []

        if not self.playing_ad:
            self.show_homepage()

    # === Ad Logic ===
    def start_ad(self, age_id, gender_id):
        if self.playing_ad:
            return
        self.playing_ad = True
        age_label = AGE_LIST[age_id].replace("(", "").replace(")", "")
        gender_label = GENDER_LIST[gender_id]
        self.scroll_text = f" Detected: {gender_label}, Age Group: {age_label} "
        self.scroll_pos = 0

        folder_path = os.path.join(ADS_DIR, age_label, gender_label)
        self.ad_playlist = []
        for ext in ("*.jpg", "*.png", "*.jpeg", "*.gif"):
            self.ad_playlist += glob.glob(os.path.join(folder_path, ext))

        if not self.ad_playlist:
            fallback_path = os.path.join(ADS_DIR, age_label)
            for ext in ("*.jpg", "*.png", "*.jpeg", "*.gif"):
                self.ad_playlist += glob.glob(os.path.join(fallback_path, ext))

        if not self.ad_playlist:
            print(f"No ad found for {age_label}/{gender_label}")
            self.playing_ad = False
            return

        random.shuffle(self.ad_playlist)
        self.ad_index = 0
        self.start_posters()

    def start_posters(self):
        if not self.playing_ad:
            return
        self.poster_timer = QtCore.QTimer()
        self.poster_timer.timeout.connect(self.show_next_poster)
        self.show_next_poster()
        self.poster_timer.start(POSTER_DISPLAY_TIME * 1000)

    def show_next_poster(self):
        if not self.playing_ad or not self.ad_playlist:
            return
        if self.ad_index >= len(self.ad_playlist):
            self.ad_index = 0
        poster_path = self.ad_playlist[self.ad_index]
        ext = os.path.splitext(poster_path)[1].lower()

        movie = self.video_label.movie()
        if movie:
            movie.stop()
        self.video_label.clear()

        try:
            if ext == ".gif":
                self.show_gif(poster_path)
            else:
                img = cv2.imread(poster_path)
                if img is not None:
                    self.update_ui_frame(img)
        except Exception as e:
            print("Error showing poster:", e)

        self.ad_index += 1

        if (time.time() - self.last_seen) > 1.5:
            self.stop_posters()

    def show_gif(self, gif_path):
        try:
            movie = QtGui.QMovie(gif_path)
            movie.setCacheMode(QtGui.QMovie.CacheAll)
            movie.start()
            while movie.frameRect().size().isEmpty():
                QtWidgets.QApplication.processEvents()
                time.sleep(0.05)
            gif_size = movie.frameRect().size()
            label_size = self.video_label.size()
            scale = min(label_size.width() / gif_size.width(),
                        label_size.height() / gif_size.height())
            movie.setScaledSize(QtCore.QSize(
                int(gif_size.width() * scale), int(gif_size.height() * scale)))
            self.video_label.setMovie(movie)
            movie.start()
        except Exception as e:
            print("Error playing GIF:", e)

    def stop_posters(self):
        if not self.playing_ad:
            return
        self.playing_ad = False
        if hasattr(self, "poster_timer") and self.poster_timer:
            self.poster_timer.stop()
            self.poster_timer = None
        movie = self.video_label.movie()
        if movie:
            movie.stop()
        self.video_label.clear()
        self.show_homepage()

    def closeEvent(self, event):
        try:
            if hasattr(self, "poster_timer") and self.poster_timer:
                self.poster_timer.stop()
        except:
            pass
        try:
            if hasattr(self, "worker"):
                self.worker.stop()
        except:
            pass
        event.accept()


# ---------- MAIN ----------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = SmartAdApp()
    window.show()
    sys.exit(app.exec_())
