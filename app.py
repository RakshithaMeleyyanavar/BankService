"""
Adaptive Reel Detox System — Flask Backend
==========================================
Main application server that handles:
- User authentication
- Webcam frame processing (OpenCV)
- Emotion detection (FER2013 model)
- Blink detection (MRL Eye model)
- Addiction risk scoring
- Detox intervention logic
- Analytics dashboard data
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
import cv2
import numpy as np
import base64
import json
import sqlite3
import os
import time
from datetime import datetime, timedelta
from utils.face_processor import FaceProcessor
from utils.addiction_scorer import AddictionScorer
from utils.detox_engine import DetoxEngine
from utils.db_manager import DBManager

app = Flask(__name__)
app.secret_key = "adaptive_reel_detox_secret_2024"
CORS(app)

# ── Initialize core AI modules ────────────────────────────────────
face_processor  = FaceProcessor()
addiction_scorer = AddictionScorer()
detox_engine    = DetoxEngine()
db_manager      = DBManager()

# ── Demo users (replace with real DB in production) ───────────────
DEMO_USERS = {
    "rakshitha": {"password": "pass123", "age": 22, "name": "Rakshitha"},
    "demo":      {"password": "demo123", "age": 17, "name": "Demo User"},
}

# ═══════════════════════════════════════════════════════════════════
# AUTH ROUTES
# ═══════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('reels'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data     = request.get_json()
        username = data.get('username', '').lower()
        password = data.get('password', '')

        if username in DEMO_USERS and DEMO_USERS[username]['password'] == password:
            session['username'] = username
            session['name']     = DEMO_USERS[username]['name']
            session['age']      = DEMO_USERS[username]['age']
            session['login_time'] = datetime.now().isoformat()
            db_manager.log_session_start(username)
            return jsonify({'success': True, 'name': DEMO_USERS[username]['name'],
                            'age': DEMO_USERS[username]['age']})
        return jsonify({'success': False, 'error': 'Invalid credentials'})

    return render_template('login.html')

@app.route('/logout')
def logout():
    if 'username' in session:
        db_manager.log_session_end(session['username'])
    session.clear()
    return redirect(url_for('login'))

# ═══════════════════════════════════════════════════════════════════
# MAIN PAGES
# ═══════════════════════════════════════════════════════════════════

@app.route('/reels')
def reels():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('reels.html',
                           username=session['username'],
                           name=session['name'],
                           age=session['age'])

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html',
                           username=session['username'],
                           name=session['name'])

# ═══════════════════════════════════════════════════════════════════
# AI PROCESSING ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    """
    Receives base64 webcam frame, runs:
    1. Face detection (OpenCV Haar cascade)
    2. Eye region extraction + blink detection
    3. Emotion classification (FER2013 CNN)
    Returns extracted features — frame is NOT stored.
    """
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    data       = request.get_json()
    frame_b64  = data.get('frame', '')
    timestamp  = data.get('timestamp', time.time())

    if not frame_b64:
        return jsonify({'error': 'No frame provided'}), 400

    try:
        # Decode base64 → OpenCV image
        img_bytes = base64.b64decode(frame_b64.split(',')[-1])
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame     = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Invalid frame'}), 400

        # ── Run AI processing pipeline ────────────────────────────
        results = face_processor.process(frame)

        # Frame is discarded here — only features stored
        del frame, img_array, img_bytes

        # ── Save behavioral features to DB ────────────────────────
        db_manager.save_frame_features(session['username'], results, timestamp)

        return jsonify({'success': True, 'features': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/scroll_event', methods=['POST'])
def scroll_event():
    """
    Receives scroll behavioral signals from frontend.
    """
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    data = request.get_json()
    scroll_data = {
        'scroll_speed':      data.get('scrollSpeed', 0),
        'swipe_frequency':   data.get('swipeFreq', 0),
        'watch_time':        data.get('watchTime', 0),
        'pause_duration':    data.get('pauseDuration', 0),
        'interaction_rate':  data.get('interactionRate', 0),
        'reel_index':        data.get('reelIndex', 0),
        'timestamp':         data.get('timestamp', time.time()),
        'hour_of_day':       datetime.now().hour,
    }

    db_manager.save_scroll_event(session['username'], scroll_data)
    return jsonify({'success': True})


@app.route('/api/get_addiction_score', methods=['POST'])
def get_addiction_score():
    """
    Aggregates recent behavioral signals and computes:
    - Compulsion Score
    - Fatigue Score
    - Emotional Volatility Score
    - Overall Addiction Risk Score
    - Detox Level (1-9)
    """
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    data         = request.get_json()
    scroll_data  = data.get('scrollData', {})
    frame_data   = data.get('frameData', {})
    age          = session.get('age', 25)

    # ── Compute scores ────────────────────────────────────────────
    scores = addiction_scorer.compute_scores(scroll_data, frame_data, age)

    # ── Determine detox level ─────────────────────────────────────
    detox_action = detox_engine.get_intervention(scores, age)

    # ── Save to DB ────────────────────────────────────────────────
    db_manager.save_addiction_score(session['username'], scores, detox_action)

    return jsonify({
        'success':      True,
        'scores':       scores,
        'detox':        detox_action,
        'timestamp':    datetime.now().isoformat()
    })


# ═══════════════════════════════════════════════════════════════════
# ANALYTICS API
# ═══════════════════════════════════════════════════════════════════

@app.route('/api/analytics/summary')
def analytics_summary():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    username = session['username']
    summary  = db_manager.get_analytics_summary(username)
    return jsonify(summary)

@app.route('/api/analytics/trends')
def analytics_trends():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    username = session['username']
    days     = int(request.args.get('days', 7))
    trends   = db_manager.get_trends(username, days)
    return jsonify(trends)

@app.route('/api/analytics/emotions')
def analytics_emotions():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    emotions = db_manager.get_emotion_distribution(session['username'])
    return jsonify(emotions)

@app.route('/api/analytics/detox_history')
def detox_history():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    history = db_manager.get_detox_history(session['username'])
    return jsonify(history)

@app.route('/api/analytics/recommendations')
def recommendations():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    recs = db_manager.get_ai_recommendations(session['username'])
    return jsonify(recs)


# ═══════════════════════════════════════════════════════════════════
# REELS DATA API
# ═══════════════════════════════════════════════════════════════════

@app.route('/api/reels')
def get_reels():
    """Returns mock reel data for the feed."""
    reels = [
        {"id": i, "title": f"Reel #{i}", "category": cat,
         "duration": 15 + (i % 4) * 15,
         "thumbnail": f"https://picsum.photos/seed/{i+100}/400/700",
         "videoUrl": "#",
         "likes": 1000 + i * 237,
         "comments": 50 + i * 13}
        for i, cat in enumerate([
            "dance","comedy","food","travel","fitness",
            "beauty","education","gaming","fashion","music",
            "dance","comedy","food","travel","fitness",
            "beauty","education","gaming","fashion","music",
        ], 1)
    ]
    return jsonify(reels)


if __name__ == '__main__':
    db_manager.init_db()
    app.run(debug=True, port=5000)
