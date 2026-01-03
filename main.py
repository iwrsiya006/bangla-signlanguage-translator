print("MAIN STARTED")
from mediapipe_utils import MediaPipeExtractor


# ================================================
# Bangla Sign Language Translator (BSL)
# STEP 1: Core Infrastructure + MediaPipe Extraction
# GUI Framework: PyQt6
# Keypoints: Hands + Pose (MediaPipe)
# ================================================

# -------------------------------
# requirements.txt (for reference)
# -------------------------------
# python>=3.9
# opencv-python
# mediapipe
# numpy
# tensorflow
# pyqt6
# scikit-learn

# -------------------------------
# mediapipe_utils.py
# -------------------------------
# This file contains all MediaPipe logic for extracting
# Hands + Pose keypoints from a video frame.

import cv2
import numpy as np
import mediapipe as mp

class MediaPipeExtractor:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Drawing utilities (for visualization later)
        self.mp_draw = mp.solutions.drawing_utils

    def extract_keypoints(self, frame):
        """
        Takes a BGR frame from OpenCV and returns a single
        flattened numpy array of Hands + Pose keypoints.
        """
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        hand_results = self.hands.process(image)
        pose_results = self.pose.process(image)

        image.flags.writeable = True

        keypoints = []

        # -------------------------------
        # Hands keypoints (2 hands max)
        # 21 landmarks per hand
        # Each landmark: x, y, z
        # -------------------------------
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])

        # If only one or zero hands detected, pad with zeros
        while len(keypoints) < 21 * 2 * 3:
            keypoints.extend([0.0, 0.0, 0.0])

        # -------------------------------
        # Pose keypoints
        # 33 landmarks
        # Each landmark: x, y, z
        # -------------------------------
        if pose_results.pose_landmarks:
            for lm in pose_results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * (33 * 3))

        return np.array(keypoints, dtype=np.float32)

    def draw_landmarks(self, frame):
        """
        Draws hands and pose landmarks on frame.
        Used by visualiser and real-time translator.
        """
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(image)
        pose_results = self.pose.process(image)

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

        if pose_results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )

        return frame

# -------------------------------
# NOTE:
# OpenCV test block REMOVED for PyQt6 mode
# This code was used only to verify MediaPipe on macOS
# It MUST NOT run together with PyQt6
# -------------------------------


# ================================================
# STEP 2: PyQt6 Main Application + Camera Feed
# This replaces cv2.imshow with a proper GUI
# Layout matches the provided reference image
# ================================================

# -------------------------------
# main.py
# -------------------------------
# Entry point for the BSL application

import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QLabel, QPushButton, QVBoxLayout, QHBoxLayout
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap

# -------------------------------
# Camera Thread (NON-BLOCKING)
# -------------------------------
# Runs OpenCV + MediaPipe in background
# Prevents GUI freezing

class CameraThread(QThread):
    frame_signal = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.running = True
        self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        self.extractor = MediaPipeExtractor()

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Draw landmarks for preview
            frame = self.extractor.draw_landmarks(frame)

            # Convert OpenCV frame (BGR) -> Qt Image (RGB)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w

            qt_image = QImage(
                rgb.data,
                w,
                h,
                bytes_per_line,
                QImage.Format.Format_RGB888
            )

            self.frame_signal.emit(qt_image)

        self.cap.release()

    def stop(self):
        self.running = False
        self.wait()

# -------------------------------
# Main Window UI
# -------------------------------

class TrainerThread(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, dataset_path='dataset'):
        super().__init__()
        self.dataset_path = dataset_path

    def run(self):
        import os, json
        import numpy as np
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Masking
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split

        MAX_LEN = 60

        X, y = [], []

        # Load dataset automatically
        for word in sorted(os.listdir(self.dataset_path)):
            word_dir = os.path.join(self.dataset_path, word)
            if not os.path.isdir(word_dir):
                continue
            for file in os.listdir(word_dir):
                if file.endswith('.npy'):
                    data = np.load(os.path.join(word_dir, file))
                    X.append(data)
                    y.append(word)

        self.progress_signal.emit(f"Loaded {len(X)} samples")

        # Pad sequences (END padding)
        def pad_sequence(seq):
            if len(seq) >= MAX_LEN:
                return seq[:MAX_LEN]
            pad = np.zeros((MAX_LEN - len(seq), seq.shape[1]))
            return np.vstack([seq, pad])

        X = np.array([pad_sequence(seq) for seq in X])

        le = LabelEncoder()
        y = le.fit_transform(y)

        with open('labels.json', 'w') as f:
            json.dump({label: int(idx) for idx, label in enumerate(le.classes_)}, f)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = Sequential([
            Masking(mask_value=0.0, input_shape=(MAX_LEN, X.shape[2])),
            LSTM(128),
            Dense(len(le.classes_), activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.progress_signal.emit("Training started...")

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=16,
            verbose=0
        )

        model.save('model.h5')
        self.progress_signal.emit("Training complete. Model saved.")
        self.finished_signal.emit()


class TranslatorThread(QThread):
    prediction_signal = pyqtSignal(str)

    def __init__(self, extractor):
        super().__init__()
        import json
        import tensorflow as tf

        self.extractor = extractor
        self.model = tf.keras.models.load_model('model.h5')
        with open('labels.json') as f:
            self.labels = {v: k for k, v in json.load(f).items()}

        self.sequence = []
        self.MAX_LEN = 60
        self.running = True

    def run(self):
        import time
        import numpy as np

        print("Translator thread started")

        while self.running:
            if len(self.sequence) >= self.MAX_LEN:
                seq = np.expand_dims(
                    np.array(self.sequence[-self.MAX_LEN:]),
                    axis=0
                )

                preds = self.model.predict(seq, verbose=0)[0]
                label = self.labels[int(np.argmax(preds))]
                confidence = float(np.max(preds))

                print("Pred:", label, "conf:", confidence)

                if confidence > 0.3:
                    self.prediction_signal.emit(label)

            time.sleep(0.03)



    def stop(self):
        self.running = False
        self.wait()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bangla Sign Language Translator")
        self.setGeometry(100, 100, 1200, 700)

        # -------------------------------
        # State variables
        # -------------------------------
        self.mode = None  # None / 'EXTRACT' / 'VISUALISE' / 'TRANSLATE'
        self.recording = False
        self.countdown_active = False
        self.recorded_frames = []
        self.word_name = ""
        self.duration_seconds = 0.0
        self.record_start_time = None
        self.visualise_data = None
        self.visualise_index = 0
        self.subtitle_sentence = []

        # Countdown timer (non-blocking)
        from PyQt6.QtCore import QTimer
        self.countdown_timer = QTimer()
        self.countdown_timer.timeout.connect(self.update_countdown)
        self.countdown_value = 0

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)

        # -------------------------------
        # Left Menu
        # -------------------------------
        self.btn_extractor = QPushButton("Extractor")
        self.btn_extractor.clicked.connect(self.start_extractor_mode)
        self.btn_visualiser = QPushButton("Visualiser")
        self.btn_visualiser.clicked.connect(self.start_visualiser_mode)
        self.btn_trainer = QPushButton("Trainer")
        self.btn_trainer.clicked.connect(self.start_trainer_mode)
        self.btn_translator = QPushButton("Translator")
        self.btn_translator.clicked.connect(self.start_translator_mode)

        for btn in [self.btn_extractor, self.btn_visualiser, self.btn_trainer, self.btn_translator]:
            btn.setFixedHeight(50)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.btn_extractor)
        left_layout.addWidget(self.btn_visualiser)
        left_layout.addWidget(self.btn_trainer)
        left_layout.addWidget(self.btn_translator)
        left_layout.addStretch()

        # -------------------------------
        # Camera Display Area
        # -------------------------------
        self.camera_label = QLabel("Camera Feed")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setStyleSheet("background-color: black; color: white;")

        # -------------------------------
        # Status / Subtitle Area
        # -------------------------------
        self.subtitle_label = QLabel("Ready")
        self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle_label.setFixedHeight(60)
        self.subtitle_label.setStyleSheet(
            "font-size: 20px; background-color: #222; color: #00ffcc;"
        )

        center_layout = QVBoxLayout()
        center_layout.addWidget(self.camera_label)
        center_layout.addWidget(self.subtitle_label)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(center_layout, 4)
        central.setLayout(main_layout)

        # -------------------------------
        # Start Camera Thread
        # -------------------------------
        self.camera_thread = CameraThread()
        self.camera_thread.frame_signal.connect(self.update_frame)
        self.camera_thread.start()

    def start_visualiser_mode(self):
        """
        STEP 4A: Visualiser Mode
        - Select a saved .npy file
        - Replay skeletons frame-by-frame
        """
        from PyQt6.QtWidgets import QFileDialog

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select recorded sample",
            "dataset",
            "NumPy files (*.npy)"
        )

        if not file_path:
            return

        self.mode = 'VISUALISE'
        self.visualise_data = np.load(file_path)
        self.visualise_index = 0

        self.subtitle_label.setText(
            f"Visualising: {file_path.split('/')[-1]}"
        )

    def start_trainer_mode(self):
        """
        STEP B: Trainer Mode
        - Automatically loads all dataset
        - Trains LSTM model
        - Shows progress in subtitle area
        """
        self.subtitle_label.setText("Starting training...")
        self.trainer_thread = TrainerThread()
        self.trainer_thread.progress_signal.connect(self.subtitle_label.setText)
        self.trainer_thread.finished_signal.connect(
            lambda: self.subtitle_label.setText("Training finished ✓")
        )
        self.trainer_thread.start()

    def start_translator_mode(self):
        """
        STEP C: Real-time Translator Mode
        - Loads trained model
        - Performs live inference
        - Displays subtitles
        """
        self.mode = 'TRANSLATE'
        self.subtitle_sentence = []
        self.subtitle_label.setText("Translator mode active")

        self.translator_thread = TranslatorThread(self.camera_thread.extractor)
        self.translator_thread.prediction_signal.connect(self.update_subtitle)
        self.translator_thread.start()

    def start_extractor_mode(self):
        """
        STEP A: Extractor Mode
        - Ask for word name + duration
        - Enable keyboard-controlled recording (press R)
        """
        from PyQt6.QtWidgets import QInputDialog

        # Reset other modes
        self.visualise_data = None
        self.visualise_index = 0

        word, ok1 = QInputDialog.getText(self, "Word Name", "Enter word name:")
        if not ok1 or word.strip() == "":
            return

        duration, ok2 = QInputDialog.getDouble(
            self,
            "Duration",
            "Enter duration (seconds):",
            decimals=1,
            min=0.5,
            max=10.0
        )
        if not ok2:
            return

        # Set extractor state
        self.mode = 'EXTRACT'
        self.word_name = word.strip()
        self.duration_seconds = duration
        self.recorded_frames = []
        self.recording = False
        self.countdown_active = False
        self.subtitle_label.setText(
            f"Extractor mode: '{self.word_name}' | Press R to record"
        )

    def update_countdown(self):
        """
        Handles the 3-second non-blocking countdown before recording starts.
        """
        import time

        self.countdown_value -= 1

        if self.countdown_value > 0:
            self.subtitle_label.setText(f"Get ready... {self.countdown_value}")
        else:
            self.countdown_timer.stop()
            self.countdown_active = False
            self.recording = True
            self.record_start_time = time.time()
            self.recorded_frames = []
            self.subtitle_label.setText(
                f"Recording '{self.word_name}'  0.0/{self.duration_seconds:.1f} sec"
            )

    def update_subtitle(self, word):
        """
        Adds stable predicted words into a sentence.
        """
        if not self.subtitle_sentence or self.subtitle_sentence[-1] != word:
            self.subtitle_sentence.append(word)
            self.subtitle_label.setText(" ".join(self.subtitle_sentence))

    def update_frame(self, image: QImage):
        print("update_frame called, self =", type(self))
        
        import cv2
        import numpy as np

    # -------------------------------
    # Translator Mode: feed keypoints
    # -------------------------------
        if self.mode == 'TRANSLATE' and hasattr(self, 'translator_thread'):
            frame = image.convertToFormat(QImage.Format.Format_RGB888)
            w, h = frame.width(), frame.height()
            ptr = frame.bits()
            ptr.setsize(h * w * 3)
            arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 3))
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

            keypoints = self.camera_thread.extractor.extract_keypoints(bgr)
            self.translator_thread.sequence.append(keypoints)

            if len(self.translator_thread.sequence) > self.translator_thread.MAX_LEN:
                self.translator_thread.sequence.pop(0)

    # -------------------------------
    # Visualiser Mode
    # -------------------------------
        if self.mode == 'VISUALISE' and self.visualise_data is not None:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)

            if self.visualise_index < len(self.visualise_data):
                frame_data = self.visualise_data[self.visualise_index]
                self.visualise_index += 1

                offset = 0
                for _ in range(2):
                    for _ in range(21):
                        x = int(frame_data[offset] * 640)
                        y = int(frame_data[offset + 1] * 480)
                        cv2.circle(blank, (x, y), 4, (0, 255, 0), -1)
                        offset += 3

                for _ in range(33):
                    x = int(frame_data[offset] * 640)
                    y = int(frame_data[offset + 1] * 480)
                    cv2.circle(blank, (x, y), 3, (255, 0, 0), -1)
                    offset += 3

                rgb = cv2.cvtColor(blank, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qt_image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
            else:
                pixmap = QPixmap(640, 480)
                pixmap.fill(Qt.GlobalColor.black)
        else:
            pixmap = QPixmap.fromImage(image)

        self.camera_label.setPixmap(
            pixmap.scaled(
                self.camera_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )

    # -------------------------------
    # Extractor Mode Recording
    # -------------------------------
        if self.mode == 'EXTRACT' and self.recording:
            import time

            elapsed = time.time() - self.record_start_time

            frame = image.convertToFormat(QImage.Format.Format_RGB888)
            w, h = frame.width(), frame.height()
            ptr = frame.bits()
            ptr.setsize(h * w * 3)
            arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 3))
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

            keypoints = self.camera_thread.extractor.extract_keypoints(bgr)
            self.recorded_frames.append(keypoints)

            self.subtitle_label.setText(
                f"Recording '{self.word_name}' {elapsed:.1f}/{self.duration_seconds:.1f} sec"
            )

            if elapsed >= self.duration_seconds:
                self.finish_recording()

    def finish_recording(self):
        import os

        self.recording = False

        os.makedirs(f"dataset/{self.word_name}", exist_ok=True)
        existing = len(os.listdir(f"dataset/{self.word_name}"))
        filename = f"dataset/{self.word_name}/sample_{existing+1:03d}.npy"

        np.save(filename, np.array(self.recorded_frames))

        self.subtitle_label.setText(
            f"Saved {os.path.basename(filename)} | Press R to record again"
        )

        self.recorded_frames = []

    def keyPressEvent(self, event):
        """
        Keyboard controls for Extractor Mode
        R → 3-second countdown → start recording
        """
        if self.mode == 'EXTRACT' and event.key() == Qt.Key.Key_R:
            if not self.recording and not self.countdown_active:
                self.countdown_value = 3
                self.countdown_active = True
                self.subtitle_label.setText(
                    f"Get ready... {self.countdown_value}"
                )
                self.countdown_timer.start(1000)
        super().keyPressEvent(event)

    def closeEvent(self, event):
        self.camera_thread.stop()
        event.accept()
    
# -------------------------------
# Application Entry Point
# -------------------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    print("MainWindow created")
    window.show()
    sys.exit(app.exec())
