"""
utils/addiction_scorer.py
=========================
Computes multi-dimensional addiction risk scores by combining:
- Scroll behavioral signals (speed, frequency, watch time)
- Facial analysis signals (blink rate, emotion, fatigue)
- Temporal signals (hour of day, session duration)

Outputs:
- compulsion_score      (0-100)
- fatigue_score         (0-100)
- emotional_volatility  (0-100)
- overall_risk          (0-100)
- risk_level            (low / moderate / high / critical)
"""

import numpy as np
import time

class AddictionScorer:

    # ── Thresholds calibrated from scroll behavior research ───────
    SCROLL_SPEED_NORMAL    = 300   # px/s normal reading
    SCROLL_SPEED_COMPULSIVE= 1200  # px/s compulsive scrolling
    SWIPE_FREQ_NORMAL      = 0.3   # swipes/sec normal
    SWIPE_FREQ_COMPULSIVE  = 1.5   # swipes/sec compulsive
    BLINK_NORMAL           = 15    # blinks/min healthy
    BLINK_FATIGUE          = 6     # blinks/min fatigue threshold
    WATCH_TIME_MAX         = 3600  # 60 min session cap

    def __init__(self):
        self._score_history = []

    def compute_scores(self, scroll_data: dict, frame_data: dict, age: int = 25) -> dict:
        """
        Main scoring function.
        Combines scroll behavior + facial signals → multi-dim risk scores.
        """
        ts = time.time()

        # ── 1. Compulsion Score ───────────────────────────────────
        compulsion = self._compute_compulsion(scroll_data)

        # ── 2. Fatigue Score ──────────────────────────────────────
        fatigue = self._compute_fatigue(frame_data, scroll_data)

        # ── 3. Emotional Volatility Score ─────────────────────────
        volatility = self._compute_volatility(frame_data)

        # ── 4. Late Night Penalty ─────────────────────────────────
        hour = scroll_data.get('hour_of_day', 12)
        night_penalty = self._night_penalty(hour)

        # ── 5. Age Adjustment ─────────────────────────────────────
        age_multiplier = 1.3 if age < 18 else (1.1 if age < 25 else 1.0)

        # ── 6. Overall Risk ───────────────────────────────────────
        overall = (
            compulsion   * 0.40 +
            fatigue      * 0.25 +
            volatility   * 0.20 +
            night_penalty* 0.15
        ) * age_multiplier

        overall = min(100, round(overall, 1))

        # ── 7. Risk Level ─────────────────────────────────────────
        if overall < 25:
            risk_level = 'low'
        elif overall < 50:
            risk_level = 'moderate'
        elif overall < 75:
            risk_level = 'high'
        else:
            risk_level = 'critical'

        # ── 8. Sub-scores ─────────────────────────────────────────
        scores = {
            'compulsion_score':      round(compulsion, 1),
            'fatigue_score':         round(fatigue, 1),
            'emotional_volatility':  round(volatility, 1),
            'night_penalty':         round(night_penalty, 1),
            'overall_risk':          overall,
            'risk_level':            risk_level,
            'blink_rate':            frame_data.get('blink_rate', 0),
            'dominant_emotion':      frame_data.get('emotion', 'neutral'),
            'watch_time_min':        round(scroll_data.get('watch_time', 0) / 60, 1),
            'scroll_speed':          scroll_data.get('scroll_speed', 0),
            'focus_score':           round(max(0, 100 - overall), 1),
            'productivity_impact':   self._productivity_impact(overall),
            'wellbeing_score':       round(max(0, 100 - overall * 0.8), 1),
            'timestamp':             ts,
        }

        self._score_history.append(scores)
        if len(self._score_history) > 100:
            self._score_history.pop(0)

        return scores

    # ── Compulsion Score ──────────────────────────────────────────
    def _compute_compulsion(self, scroll_data: dict) -> float:
        speed     = scroll_data.get('scroll_speed', 0)
        freq      = scroll_data.get('swipe_frequency', 0)
        watch     = scroll_data.get('watch_time', 0)
        pause     = scroll_data.get('pause_duration', 5)
        interact  = scroll_data.get('interaction_rate', 0)

        # Speed score: high speed = compulsive scrolling
        speed_score = min(100, (speed / self.SCROLL_SPEED_COMPULSIVE) * 100)

        # Frequency score
        freq_score  = min(100, (freq / self.SWIPE_FREQ_COMPULSIVE) * 100)

        # Watch time score
        watch_score = min(100, (watch / self.WATCH_TIME_MAX) * 100)

        # Short pause = not absorbing = compulsive
        pause_score = max(0, 100 - (pause / 30) * 100)

        # Low interaction = zombie scrolling
        interact_score = max(0, 80 - interact * 400)

        return (
            speed_score    * 0.30 +
            freq_score     * 0.25 +
            watch_score    * 0.20 +
            pause_score    * 0.15 +
            interact_score * 0.10
        )

    # ── Fatigue Score ─────────────────────────────────────────────
    def _compute_fatigue(self, frame_data: dict, scroll_data: dict) -> float:
        blink_rate     = frame_data.get('blink_rate', 15)
        ear            = frame_data.get('ear', 0.3)
        fatigue_signal = frame_data.get('fatigue_signal', 0)
        watch_time     = scroll_data.get('watch_time', 0)

        # Low blink rate = eye fatigue
        blink_score  = max(0, min(100,
            (1 - blink_rate / self.BLINK_NORMAL) * 100
        )) if blink_rate < self.BLINK_NORMAL else 0

        # Low EAR = droopy eyes
        ear_score    = max(0, (1 - ear / 0.3) * 100) if ear > 0 else 50

        # Cumulative watch time
        time_score   = min(100, (watch_time / 7200) * 100)  # 2h max

        # Face processor fatigue signal
        fs_score     = fatigue_signal * 100

        return (
            blink_score * 0.35 +
            ear_score   * 0.25 +
            time_score  * 0.20 +
            fs_score    * 0.20
        )

    # ── Emotional Volatility Score ────────────────────────────────
    def _compute_volatility(self, frame_data: dict) -> float:
        emotion  = frame_data.get('emotion', 'neutral')
        scores   = frame_data.get('emotion_scores', {})

        # Negative emotions increase volatility
        negative_weight = {
            'angry': 1.0, 'fear': 0.9, 'disgust': 0.8,
            'sad': 0.7, 'surprise': 0.4, 'neutral': 0.1, 'happy': 0.0
        }
        vol = negative_weight.get(emotion, 0.3) * 60

        # Add entropy of emotion distribution (high entropy = volatile)
        if scores:
            vals = np.array(list(scores.values()))
            vals = vals / (vals.sum() + 1e-9)
            entropy = -np.sum(vals * np.log(vals + 1e-9))
            vol += entropy * 10  # scale to 0-20

        return min(100, vol)

    # ── Late Night Penalty ────────────────────────────────────────
    def _night_penalty(self, hour: int) -> float:
        """Usage after 10pm or before 6am increases risk."""
        if 22 <= hour or hour < 6:
            return 70.0
        elif 20 <= hour < 22:
            return 30.0
        return 0.0

    # ── Productivity Impact ───────────────────────────────────────
    def _productivity_impact(self, overall_risk: float) -> str:
        if overall_risk < 25:   return 'Minimal'
        elif overall_risk < 50: return 'Low'
        elif overall_risk < 75: return 'Moderate'
        else:                   return 'High'
