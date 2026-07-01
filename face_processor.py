"""
utils/face_processor.py
=======================
Digital Image Processing module using OpenCV.

Performs:
1. Face detection (Haar Cascade)
2. Eye region extraction
3. Blink detection (EAR — Eye Aspect Ratio)
4. Gaze direction estimation
5. Head tilt detection
6. Emotion classification (FER2013 CNN)

Privacy: Raw frames are NEVER stored. Only numerical features returned.
"""

import cv2
import numpy as np
import os

# ── Try importing TensorFlow ──────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ── Emotion labels (FER2013 order) ───────────────────────────────
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ── EAR threshold for blink detection ────────────────────────────
EAR_THRESHOLD   = 0.25
BLINK_CONSEC    = 2

class FaceProcessor:
    """
    Main computer vision pipeline for real-time behavioral analysis.
    Uses OpenCV for image processing and a CNN for emotion detection.
    """

    def __init__(self):
        self.face_cascade = None
        self.eye_cascade  = None
        self.emotion_model = None
        self._load_cascades()
        self._load_emotion_model()
        self.blink_counter  = 0
        self.blink_total    = 0
        self.frame_count    = 0
        self.session_start  = __import__('time').time()

    # ── Load OpenCV cascade classifiers ──────────────────────────
    def _load_cascades(self):
        cv2_data = cv2.data.haarcascades
        face_path = os.path.join(cv2_data, 'haarcascade_frontalface_default.xml')
        eye_path  = os.path.join(cv2_data, 'haarcascade_eye.xml')

        if os.path.exists(face_path):
            self.face_cascade = cv2.CascadeClassifier(face_path)
        if os.path.exists(eye_path):
            self.eye_cascade = cv2.CascadeClassifier(eye_path)

    # ── Load pre-trained emotion model ───────────────────────────
    def _load_emotion_model(self):
        model_path = 'models/emotion_model.h5'
        if TF_AVAILABLE and os.path.exists(model_path):
            try:
                self.emotion_model = load_model(model_path)
            except Exception as e:
                print(f"[FaceProcessor] Could not load emotion model: {e}")
                self.emotion_model = None
        else:
            self.emotion_model = None

    # ── Main processing pipeline ─────────────────────────────────
    def process(self, frame):
        """
        Process a single video frame.
        Returns dict of extracted features.
        Frame is never stored — only numbers returned.
        """
        self.frame_count += 1
        results = {
            'face_detected':   False,
            'eye_count':       0,
            'blink_detected':  False,
            'blink_rate':      0.0,   # blinks per minute
            'ear':             0.0,   # Eye Aspect Ratio
            'emotion':         'neutral',
            'emotion_scores':  {e: 0.0 for e in EMOTIONS},
            'gaze_direction':  'center',
            'head_tilt':       0.0,
            'face_area_ratio': 0.0,   # proxy for distance from screen
            'fatigue_signal':  0.0,
        }

        if frame is None:
            return results

        # Convert to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        if self.face_cascade is None:
            return results

        # ── Step 1: Face Detection ────────────────────────────────
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        if len(faces) == 0:
            return results

        results['face_detected'] = True

        # Use the largest detected face
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        results['face_area_ratio'] = round((fw * fh) / (w * h), 3)

        # ── Step 2: Head Tilt Estimation ──────────────────────────
        # Tilt estimated from face bounding box aspect ratio skew
        results['head_tilt'] = round(abs(fw - fh) / max(fw, fh) * 45, 2)

        # ── Step 3: Eye Detection & EAR ───────────────────────────
        face_roi_gray = gray[fy:fy+fh, fx:fx+fw]

        if self.eye_cascade is not None:
            eyes = self.eye_cascade.detectMultiScale(
                face_roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
            )
            results['eye_count'] = len(eyes)

            if len(eyes) >= 2:
                # Compute simplified EAR from eye bounding boxes
                ear = self._compute_ear_from_boxes(eyes)
                results['ear'] = round(ear, 3)

                # Blink detection
                if ear < EAR_THRESHOLD:
                    self.blink_counter += 1
                else:
                    if self.blink_counter >= BLINK_CONSEC:
                        self.blink_total += 1
                        results['blink_detected'] = True
                    self.blink_counter = 0

                # Blink rate (blinks per minute)
                elapsed_min = max((
                    __import__('time').time() - self.session_start) / 60, 0.01
                )
                results['blink_rate'] = round(self.blink_total / elapsed_min, 2)

            # ── Step 4: Gaze Direction ────────────────────────────
            if len(eyes) > 0:
                results['gaze_direction'] = self._estimate_gaze(
                    eyes, face_roi_gray.shape[1]
                )

        # ── Step 5: Emotion Detection ─────────────────────────────
        emotion, scores = self._detect_emotion(face_roi_gray, fw, fh)
        results['emotion']        = emotion
        results['emotion_scores'] = scores

        # ── Step 6: Fatigue Signal ────────────────────────────────
        # Low blink rate + low EAR + sad/neutral emotion → fatigue
        blink_fatigue   = max(0, 1 - (results['blink_rate'] / 15))  # normal=15/min
        ear_fatigue     = max(0, 1 - (results['ear'] / EAR_THRESHOLD)) if results['ear'] > 0 else 0
        emotion_fatigue = scores.get('sad', 0) + scores.get('neutral', 0) * 0.3
        results['fatigue_signal'] = round(
            (blink_fatigue * 0.4 + ear_fatigue * 0.4 + emotion_fatigue * 0.2), 3
        )

        return results

    # ── EAR from bounding boxes (simplified) ─────────────────────
    def _compute_ear_from_boxes(self, eyes):
        """
        Simplified Eye Aspect Ratio from detected eye bounding boxes.
        Normal EAR ≈ 0.3. Blink EAR < 0.25.
        """
        ears = []
        for (ex, ey, ew, eh) in eyes[:2]:
            ear = eh / (ew + 1e-6)
            ears.append(ear)
        return np.mean(ears) if ears else 0.3

    # ── Gaze direction estimation ─────────────────────────────────
    def _estimate_gaze(self, eyes, face_width):
        """Estimate gaze direction from eye positions within face."""
        if len(eyes) == 0:
            return 'center'
        centers = [(ex + ew//2) for (ex,ey,ew,eh) in eyes]
        avg_x   = np.mean(centers) / (face_width + 1e-6)
        if avg_x < 0.35:
            return 'left'
        elif avg_x > 0.65:
            return 'right'
        return 'center'

    # ── Emotion detection ─────────────────────────────────────────
    def _detect_emotion(self, face_roi_gray, fw, fh):
        """
        Classify emotion using FER2013-trained CNN.
        Falls back to rule-based estimation if model unavailable.
        """
        scores = {e: 0.0 for e in EMOTIONS}

        if self.emotion_model is not None and TF_AVAILABLE:
            try:
                # Preprocess: resize to 48×48, normalize
                face_img = cv2.resize(face_roi_gray, (48, 48))
                face_img = face_img.astype('float32') / 255.0
                face_img = np.expand_dims(face_img, axis=(0, -1))  # (1,48,48,1)

                preds = self.emotion_model.predict(face_img, verbose=0)[0]
                scores = {EMOTIONS[i]: round(float(preds[i]), 3) for i in range(len(EMOTIONS))}
                emotion = EMOTIONS[np.argmax(preds)]
                return emotion, scores
            except Exception as e:
                pass

        # ── Rule-based fallback ───────────────────────────────────
        # Uses pixel intensity variance as crude fatigue proxy
        variance = float(np.var(face_roi_gray))
        if variance < 500:
            dominant = 'neutral'
        elif variance < 1000:
            dominant = 'sad'
        else:
            dominant = 'happy'

        scores[dominant] = 0.7
        scores['neutral'] = 0.3 if dominant != 'neutral' else 0.7
        return dominant, scores

    def reset_session(self):
        self.blink_counter = 0
        self.blink_total   = 0
        self.frame_count   = 0
        self.session_start = __import__('time').time()
