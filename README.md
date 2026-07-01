# 🧠 Adaptive Reel Detox System

An AI-powered digital wellbeing application that monitors reel consumption patterns
and automatically triggers detox interventions when addiction risk is detected.

---

## 📁 Project Structure

```
adaptive_reel_detox/
├── app.py                          # Flask backend
├── requirements.txt
├── templates/
│   ├── login.html                  # Instagram-style login
│   ├── reels.html                  # Reel feed + live monitoring
│   └── dashboard.html              # Analytics dashboard
├── utils/
│   ├── face_processor.py           # OpenCV face/eye/emotion pipeline
│   ├── addiction_scorer.py         # Multi-signal risk scoring
│   ├── detox_engine.py             # 9-level intervention system
│   └── db_manager.py               # SQLite storage (features only)
├── models/
│   ├── train_emotion_model.py      # FER2013 CNN training
│   ├── train_blink_model.py        # MRL Eye CNN training
│   └── emotion_model.h5            # (generated after training)
├── data/
│   └── reel_detox.db               # SQLite database (auto-created)
└── static/
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the application
```bash
python app.py
```

### 3. Open in browser
```
http://localhost:5000
```

### 4. Login with demo accounts
| Username    | Password  | Age | Notes              |
|-------------|-----------|-----|--------------------|
| `rakshitha` | `pass123` | 22  | Adult account      |
| `demo`      | `demo123` | 17  | Child protection active |

---

## 🤖 AI Model Training (Optional)

The app works without trained models (uses rule-based fallback).
To enable full deep learning:

### Train Emotion Model (FER2013)
```bash
# Download FER2013 from Kaggle first
python models/train_emotion_model.py \
    --data_dir /content/datasets/facial_expressions/fer2013 \
    --epochs 30 \
    --output models/emotion_model.h5
```

### Train Blink Model (MRL Eye)
```bash
python models/train_blink_model.py \
    --data_dir /content/datasets/eye_blink/mrl_eye \
    --epochs 20 \
    --output models/blink_model.h5
```

---

## 🛡️ 9-Level Detox System

| Level | Name                     | Trigger                        | Action              |
|-------|--------------------------|--------------------------------|---------------------|
| 1     | Gentle Awareness         | Risk > 15                      | Notification toast  |
| 2     | Behavioral Nudge         | Risk > 25 or 20+ min           | 30s pause + tips    |
| 3     | Adaptive Slowdown        | Risk > 40                      | Slow autoplay       |
| 4     | Cognitive Detox          | Risk > 55                      | 2 min mindfulness   |
| 5     | Temporary Lock           | Risk > 70 or 60+ min           | 15 min lock         |
| 6     | Eye Health Protection    | Blink rate < 6/min             | Eye exercise        |
| 7     | Child Protection         | Age < 18 + 60+ min             | 1 hour lock         |
| 8     | Mental Wellness          | Stress emotion detected        | 5 min breathing     |
| 9     | Hard Detox Mode          | Risk > 85 or 90+ min           | 1 hour hard lock    |

---

## 📊 Analytics Dashboard Features

- **Risk Score Trend** — line chart over 7 days
- **Emotion Distribution** — doughnut chart
- **Blink Rate Trend** — with healthy baseline
- **Fatigue Score** — color-coded bar chart
- **Wellbeing Score** — gauge (0-100)
- **Scrolling Intensity Heatmap** — session grid
- **Detox Intervention Timeline** — history log
- **AI Recommendations** — personalized insights
- **Weekly Detox Report** — stacked bar chart

---

## 🔒 Privacy

- Raw webcam frames are **never stored**
- Only numerical features extracted per frame
- Frame buffer deleted immediately after processing
- Only behavioral metrics saved to SQLite

---

## 🧪 Tech Stack

| Layer       | Technology                    |
|-------------|-------------------------------|
| Frontend    | HTML5, CSS3, Vanilla JS       |
| Backend     | Python Flask                  |
| CV Pipeline | OpenCV (Haar Cascades)        |
| Emotion AI  | TensorFlow/Keras CNN (FER2013)|
| Blink AI    | TensorFlow/Keras CNN (MRL Eye)|
| Database    | SQLite (behavioral features)  |
| Charts      | Chart.js 4.x                  |
| Fonts       | Google Fonts (Syne + DM Sans) |
