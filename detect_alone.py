# ===========================================================================
# STABLE WORKING VERSION
# ===========================================================================



import cv2
import numpy as np
import time
from collections import deque, Counter

# ---------------- Configuration ----------------

FACE_PROTO = "opencv_face_detector.pbtxt"
FACE_MODEL = "opencv_face_detector_uint8.pb"

AGE_PROTO = "age_deploy.prototxt"
AGE_MODEL = "age_net.caffemodel"

GENDER_PROTO = "gender_deploy.prototxt"
GENDER_MODEL = "gender_net.caffemodel"

# Mean values for model normalization

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Labels

AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
'(25-32)', '(38-43)', '(48-53)', '(60- 100)']
GENDER_LIST = ['Male', 'Female']

# Thresholds

FACE_CONF_THRESHOLD = 0.7
PADDING = 20
CONFIDENCE_THRESHOLD = 0.5  # for age/gender predictions

# Temporal smoothing

MAX_HISTORY = 5
prediction_history = {}

# ---------------- Load Models ----------------``

print("Loading models...")
face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
print("✅ Models loaded successfully!")

# ---------------- Helper Functions ----------------

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

def enhance_lighting(face):
    """Apply CLAHE for lighting normalization."""
    face_ycrcb = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(face_ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y = clahe.apply(y)
    face_clahe = cv2.merge((y, cr, cb))
    return cv2.cvtColor(face_clahe, cv2.COLOR_YCrCb2BGR)

def predict_age_gender(face):
    """Predict age and gender with ensemble (original + flipped)."""
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


# ---------------- Main Loop ----------------

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Failed to open webcam.")
    exit()

print("Press 'q' to quit.")
fps_start = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.05)
        continue

    #frame = cv2.flip(frame, 1)
    result_img = frame.copy()

    faces = detect_faces(face_net, frame, FACE_CONF_THRESHOLD)

    for (x1, y1, x2, y2) in faces:
        face = frame[max(0, y1 - PADDING):min(y2 + PADDING, frame.shape[0] - 1),
                    max(0, x1 - PADDING):min(x2 + PADDING, frame.shape[1] - 1)]

        if face.size == 0:
            continue

        gender_label, gender_conf, age_label, age_conf = predict_age_gender(face)

        # Filter low-confidence results
        if gender_conf < CONFIDENCE_THRESHOLD or age_conf < CONFIDENCE_THRESHOLD:
            continue

        # Temporal smoothing
        face_key = (x1 // 10, y1 // 10)
        if face_key not in prediction_history:
            prediction_history[face_key] = {'gender': deque(maxlen=MAX_HISTORY),
                                            'age': deque(maxlen=MAX_HISTORY)}

        prediction_history[face_key]['gender'].append(gender_label)
        prediction_history[face_key]['age'].append(age_label)

        gender_label = Counter(prediction_history[face_key]['gender']).most_common(1)[0][0]
        age_label = Counter(prediction_history[face_key]['age']).most_common(1)[0][0]

        label = f"{gender_label}, {age_label}"
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(result_img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

    # ---------------- FPS Calculation ----------------
    frame_count += 1
    if (time.time() - fps_start) > 1:
        fps = frame_count / (time.time() - fps_start)
        fps_start = time.time()
        frame_count = 0
        cv2.putText(result_img, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Age + Gender Detection (Enhanced)", result_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()










