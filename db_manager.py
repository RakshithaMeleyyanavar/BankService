"""
utils/db_manager.py
===================
SQLite database manager.
Stores only extracted behavioral features — NO raw video/images.

Tables:
- sessions        : login/logout tracking
- frame_features  : per-frame facial analysis results
- scroll_events   : scroll behavioral signals
- addiction_scores: computed risk scores
- detox_events    : triggered interventions
"""

import sqlite3
import json
import time
import random
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = 'data/reel_detox.db'

class DBManager:

    def init_db(self):
        """Create all tables if they don't exist."""
        Path('data').mkdir(exist_ok=True)
        with sqlite3.connect(DB_PATH) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    username    TEXT NOT NULL,
                    start_time  REAL,
                    end_time    REAL,
                    duration    REAL
                );

                CREATE TABLE IF NOT EXISTS frame_features (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    username        TEXT NOT NULL,
                    timestamp       REAL,
                    face_detected   INTEGER,
                    blink_rate      REAL,
                    ear             REAL,
                    emotion         TEXT,
                    emotion_scores  TEXT,
                    gaze_direction  TEXT,
                    head_tilt       REAL,
                    fatigue_signal  REAL
                );

                CREATE TABLE IF NOT EXISTS scroll_events (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    username        TEXT NOT NULL,
                    timestamp       REAL,
                    scroll_speed    REAL,
                    swipe_frequency REAL,
                    watch_time      REAL,
                    pause_duration  REAL,
                    interaction_rate REAL,
                    reel_index      INTEGER,
                    hour_of_day     INTEGER
                );

                CREATE TABLE IF NOT EXISTS addiction_scores (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    username            TEXT NOT NULL,
                    timestamp           REAL,
                    compulsion_score    REAL,
                    fatigue_score       REAL,
                    emotional_volatility REAL,
                    overall_risk        REAL,
                    risk_level          TEXT,
                    wellbeing_score     REAL
                );

                CREATE TABLE IF NOT EXISTS detox_events (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    username        TEXT NOT NULL,
                    timestamp       REAL,
                    level           INTEGER,
                    name            TEXT,
                    trigger_score   REAL,
                    action          TEXT
                );
            """)
            # Seed demo data so dashboard works immediately
            self._seed_demo_data()

    # ── Session tracking ──────────────────────────────────────────
    def log_session_start(self, username):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO sessions (username, start_time) VALUES (?,?)",
                (username, time.time())
            )

    def log_session_end(self, username):
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(
                "SELECT id, start_time FROM sessions WHERE username=? ORDER BY id DESC LIMIT 1",
                (username,)
            ).fetchone()
            if row:
                duration = time.time() - row[1]
                conn.execute(
                    "UPDATE sessions SET end_time=?, duration=? WHERE id=?",
                    (time.time(), duration, row[0])
                )

    # ── Feature storage ───────────────────────────────────────────
    def save_frame_features(self, username, features, timestamp):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO frame_features
                (username,timestamp,face_detected,blink_rate,ear,emotion,
                 emotion_scores,gaze_direction,head_tilt,fatigue_signal)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                username, timestamp,
                int(features.get('face_detected', False)),
                features.get('blink_rate', 0),
                features.get('ear', 0),
                features.get('emotion', 'neutral'),
                json.dumps(features.get('emotion_scores', {})),
                features.get('gaze_direction', 'center'),
                features.get('head_tilt', 0),
                features.get('fatigue_signal', 0),
            ))

    def save_scroll_event(self, username, scroll_data):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO scroll_events
                (username,timestamp,scroll_speed,swipe_frequency,watch_time,
                 pause_duration,interaction_rate,reel_index,hour_of_day)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (
                username, scroll_data.get('timestamp', time.time()),
                scroll_data.get('scroll_speed', 0),
                scroll_data.get('swipe_frequency', 0),
                scroll_data.get('watch_time', 0),
                scroll_data.get('pause_duration', 0),
                scroll_data.get('interaction_rate', 0),
                scroll_data.get('reel_index', 0),
                scroll_data.get('hour_of_day', 12),
            ))

    def save_addiction_score(self, username, scores, detox_action):
        ts = time.time()
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO addiction_scores
                (username,timestamp,compulsion_score,fatigue_score,
                 emotional_volatility,overall_risk,risk_level,wellbeing_score)
                VALUES (?,?,?,?,?,?,?,?)
            """, (
                username, ts,
                scores.get('compulsion_score', 0),
                scores.get('fatigue_score', 0),
                scores.get('emotional_volatility', 0),
                scores.get('overall_risk', 0),
                scores.get('risk_level', 'low'),
                scores.get('wellbeing_score', 100),
            ))
            if detox_action.get('level', 0) > 0:
                conn.execute("""
                    INSERT INTO detox_events
                    (username,timestamp,level,name,trigger_score,action)
                    VALUES (?,?,?,?,?,?)
                """, (
                    username, ts,
                    detox_action.get('level', 0),
                    detox_action.get('name', ''),
                    detox_action.get('trigger_score', 0),
                    detox_action.get('action', ''),
                ))

    # ── Analytics queries ─────────────────────────────────────────
    def get_analytics_summary(self, username):
        with sqlite3.connect(DB_PATH) as conn:
            # Latest risk score
            score_row = conn.execute(
                "SELECT overall_risk, risk_level, wellbeing_score, compulsion_score, fatigue_score "
                "FROM addiction_scores WHERE username=? ORDER BY timestamp DESC LIMIT 1",
                (username,)
            ).fetchone()

            # Today's usage (in minutes)
            today_start = datetime.now().replace(hour=0, minute=0, second=0).timestamp()
            session_row = conn.execute(
                "SELECT SUM(duration) FROM sessions WHERE username=? AND start_time>?",
                (username, today_start)
            ).fetchone()

            # Total detox interventions today
            detox_count = conn.execute(
                "SELECT COUNT(*) FROM detox_events WHERE username=? AND timestamp>?",
                (username, today_start)
            ).fetchone()[0]

            # Average blink rate
            blink_row = conn.execute(
                "SELECT AVG(blink_rate) FROM frame_features WHERE username=? AND blink_rate>0",
                (username,)
            ).fetchone()

            return {
                'overall_risk':      score_row[0] if score_row else 0,
                'risk_level':        score_row[1] if score_row else 'low',
                'wellbeing_score':   score_row[2] if score_row else 100,
                'compulsion_score':  score_row[3] if score_row else 0,
                'fatigue_score':     score_row[4] if score_row else 0,
                'today_usage_min':   round((session_row[0] or 0) / 60, 1),
                'detox_count_today': detox_count,
                'avg_blink_rate':    round(blink_row[0] or 15, 1),
            }

    def get_trends(self, username, days=7):
        with sqlite3.connect(DB_PATH) as conn:
            since = (datetime.now() - timedelta(days=days)).timestamp()
            rows = conn.execute(
                "SELECT timestamp, overall_risk, fatigue_score, compulsion_score "
                "FROM addiction_scores WHERE username=? AND timestamp>? ORDER BY timestamp",
                (username, since)
            ).fetchall()

            blink_rows = conn.execute(
                "SELECT timestamp, blink_rate FROM frame_features "
                "WHERE username=? AND timestamp>? AND blink_rate>0 ORDER BY timestamp",
                (username, since)
            ).fetchall()

            return {
                'risk_trend':  [
                    {'t': r[0], 'v': r[1]} for r in rows
                ],
                'fatigue_trend': [
                    {'t': r[0], 'v': r[2]} for r in rows
                ],
                'blink_trend': [
                    {'t': r[0], 'v': r[1]} for r in blink_rows
                ],
            }

    def get_emotion_distribution(self, username):
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(
                "SELECT emotion, COUNT(*) FROM frame_features "
                "WHERE username=? AND face_detected=1 GROUP BY emotion",
                (username,)
            ).fetchall()
            return {r[0]: r[1] for r in rows} if rows else {
                'neutral': 45, 'happy': 25, 'sad': 10,
                'angry': 8, 'fear': 5, 'surprise': 7
            }

    def get_detox_history(self, username):
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(
                "SELECT timestamp, level, name, trigger_score, action "
                "FROM detox_events WHERE username=? ORDER BY timestamp DESC LIMIT 50",
                (username,)
            ).fetchall()
            return [
                {'timestamp': r[0], 'level': r[1], 'name': r[2],
                 'trigger_score': r[3], 'action': r[4]}
                for r in rows
            ]

    def get_ai_recommendations(self, username):
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(
                "SELECT overall_risk, fatigue_score, emotional_volatility "
                "FROM addiction_scores WHERE username=? ORDER BY timestamp DESC LIMIT 20",
                (username,)
            ).fetchall()

        recs = []
        if rows:
            avg_risk    = sum(r[0] for r in rows) / len(rows)
            avg_fatigue = sum(r[1] for r in rows) / len(rows)
            avg_vol     = sum(r[2] for r in rows) / len(rows)

            if avg_risk > 60:
                recs.append("⚠️ Your addiction risk is consistently high. Set a 30-min daily limit.")
            if avg_fatigue > 50:
                recs.append("👁️ Eye fatigue is frequent. Use the 20-20-20 rule every session.")
            if avg_vol > 40:
                recs.append("😔 Emotional volatility detected. Consider limiting sad/stressful content.")

        if not recs:
            recs = [
                "✅ You have healthy scrolling habits. Keep it up!",
                "💡 Try the Pomodoro method: 25 min on, 5 min off.",
                "🌟 Your wellbeing score is great. Maintain this routine!",
            ]
        return recs

    # ── Seed demo data for dashboard ─────────────────────────────
    def _seed_demo_data(self):
        with sqlite3.connect(DB_PATH) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM addiction_scores"
            ).fetchone()[0]
            if count > 0:
                return  # already seeded

        with sqlite3.connect(DB_PATH) as conn:
            now = time.time()
            emotions = ['happy','neutral','sad','angry','fear','surprise']
            risk_levels = ['low','moderate','high','critical']

            for i in range(200):
                t = now - (200 - i) * 1800  # every 30 min for ~4 days
                risk = max(0, min(100, 30 + random.gauss(0, 20) + (i % 24) * 0.8))
                conn.execute("""
                    INSERT INTO addiction_scores
                    (username,timestamp,compulsion_score,fatigue_score,
                     emotional_volatility,overall_risk,risk_level,wellbeing_score)
                    VALUES ('rakshitha',?,?,?,?,?,?,?)
                """, (
                    t,
                    max(0, risk * 0.9 + random.gauss(0,5)),
                    max(0, risk * 0.7 + random.gauss(0,8)),
                    max(0, risk * 0.5 + random.gauss(0,6)),
                    risk,
                    risk_levels[min(3, int(risk/25))],
                    max(0, 100 - risk * 0.8),
                ))

                conn.execute("""
                    INSERT INTO frame_features
                    (username,timestamp,face_detected,blink_rate,ear,emotion,
                     emotion_scores,gaze_direction,head_tilt,fatigue_signal)
                    VALUES ('rakshitha',?,1,?,?,?,?,?,?,?)
                """, (
                    t,
                    max(2, 15 - risk/10 + random.gauss(0,2)),
                    max(0.1, 0.3 - risk/1000 + random.gauss(0,0.02)),
                    random.choice(emotions),
                    '{}',
                    random.choice(['center','left','right']),
                    random.uniform(0, 15),
                    min(1, risk/100),
                ))

                if risk > 40:
                    level = min(9, int(risk/12))
                    conn.execute("""
                        INSERT INTO detox_events
                        (username,timestamp,level,name,trigger_score,action)
                        VALUES ('rakshitha',?,?,?,?,?)
                    """, (t, level, f'Level {level}', risk, 'show_notification'))

                conn.execute("""
                    INSERT INTO scroll_events
                    (username,timestamp,scroll_speed,swipe_frequency,watch_time,
                     pause_duration,interaction_rate,reel_index,hour_of_day)
                    VALUES ('rakshitha',?,?,?,?,?,?,?,?)
                """, (
                    t,
                    random.uniform(200, 1500),
                    random.uniform(0.1, 2.0),
                    i * 180,
                    random.uniform(2, 30),
                    random.uniform(0, 0.2),
                    i % 50,
                    int((t % 86400) / 3600),
                ))
