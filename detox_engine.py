"""
utils/detox_engine.py
=====================
Adaptive Reel Detox Engine
Implements 9-level intervention system based on addiction risk scores.

Levels:
 1 - Gentle Awareness
 2 - Behavioral Nudge
 3 - Adaptive Slowdown
 4 - Cognitive Detox Mode
 5 - Temporary Lock Mode
 6 - Eye Health Protection
 7 - Child Protection Mode (under 18)
 8 - Mental Wellness Intervention
 9 - Hard Detox Mode
"""

from datetime import datetime

class DetoxEngine:

    INTERVENTIONS = {
        1: {
            'level':   1,
            'name':    'Gentle Awareness',
            'color':   '#22c55e',
            'icon':    '💚',
            'message': "You've been scrolling for a while. Consider taking a short break!",
            'action':  'show_notification',
            'lock_duration': 0,
            'tips': ["Take a deep breath", "Look away from screen for 20 seconds"],
        },
        2: {
            'level':   2,
            'name':    'Behavioral Nudge',
            'color':   '#84cc16',
            'icon':    '🌿',
            'message': "Time for a quick stretch! Stand up and move around for 2 minutes.",
            'action':  'pause_and_nudge',
            'lock_duration': 30,  # 30 sec pause
            'tips': ["Stretch your neck", "Drink some water", "Roll your shoulders"],
        },
        3: {
            'level':   3,
            'name':    'Adaptive Slowdown',
            'color':   '#eab308',
            'icon':    '🟡',
            'message': "Slowing down reel autoplay to help you be more mindful.",
            'action':  'slow_autoplay',
            'lock_duration': 0,
            'tips': ["Notice how each reel makes you feel", "Ask: do I really need more?"],
        },
        4: {
            'level':   4,
            'name':    'Cognitive Detox Mode',
            'color':   '#f97316',
            'icon':    '🧘',
            'message': "Reels paused. Try a quick mindfulness exercise.",
            'action':  'pause_mindfulness',
            'lock_duration': 120,  # 2 min
            'tips': [
                "Take 5 deep breaths",
                "Name 3 things you can see around you",
                "Feel your feet on the ground"
            ],
        },
        5: {
            'level':   5,
            'name':    'Temporary Lock',
            'color':   '#ef4444',
            'icon':    '🔒',
            'message': "Reels locked for 15 minutes. Use this time to recharge.",
            'action':  'lock_feed',
            'lock_duration': 900,  # 15 min
            'tips': ["Go for a short walk", "Call a friend", "Have a glass of water"],
        },
        6: {
            'level':   6,
            'name':    'Eye Health Protection',
            'color':   '#06b6d4',
            'icon':    '👁️',
            'message': "Low blink rate detected! Eye fatigue alert. Rest your eyes now.",
            'action':  'eye_protection',
            'lock_duration': 60,
            'tips': [
                "20-20-20 Rule: Look at something 20 feet away for 20 seconds",
                "Blink rapidly 10 times",
                "Close eyes gently for 30 seconds",
                "Adjust screen brightness"
            ],
        },
        7: {
            'level':   7,
            'name':    'Child Protection Mode',
            'color':   '#8b5cf6',
            'icon':    '🛡️',
            'message': "Daily reel limit reached. Parental controls active.",
            'action':  'child_protection',
            'lock_duration': 3600,  # 1 hour
            'tips': [
                "Daily limit: 60 minutes for under-18",
                "Consider calling your parents",
                "Try outdoor activities"
            ],
        },
        8: {
            'level':   8,
            'name':    'Mental Wellness Intervention',
            'color':   '#ec4899',
            'icon':    '💜',
            'message': "Emotional stress detected. We care about your wellbeing.",
            'action':  'wellness_intervention',
            'lock_duration': 300,  # 5 min
            'tips': [
                "Try box breathing: inhale 4s, hold 4s, exhale 4s",
                "You are not your emotions",
                "Talk to someone you trust"
            ],
        },
        9: {
            'level':   9,
            'name':    'Hard Detox Mode',
            'color':   '#1e293b',
            'icon':    '🚫',
            'message': "Extreme usage detected. Reels locked for 1 hour for your wellbeing.",
            'action':  'hard_lock',
            'lock_duration': 3600,  # 1 hour
            'tips': [
                "This is a sign to step away",
                "Your wellbeing is more important",
                "Try journaling how you feel right now"
            ],
        },
    }

    def get_intervention(self, scores: dict, age: int = 25) -> dict:
        """
        Determine appropriate detox level based on scores.
        Returns full intervention action object.
        """
        overall    = scores.get('overall_risk', 0)
        fatigue    = scores.get('fatigue_score', 0)
        blink_rate = scores.get('blink_rate', 15)
        emotion    = scores.get('dominant_emotion', 'neutral')
        volatility = scores.get('emotional_volatility', 0)
        watch_min  = scores.get('watch_time_min', 0)

        # ── Determine level ───────────────────────────────────────
        level = 0

        # Level 9: Extreme risk
        if overall >= 85 or watch_min > 90:
            level = 9

        # Level 8: Mental wellness (stress detected)
        elif emotion in ['angry', 'sad', 'fear'] and volatility > 60:
            level = 8

        # Level 7: Child protection
        elif age < 18 and watch_min > 60:
            level = 7

        # Level 6: Eye health (low blink rate)
        elif blink_rate < 6 and blink_rate > 0:
            level = 6

        # Level 5: High risk lock
        elif overall >= 70 or watch_min > 60:
            level = 5

        # Level 4: Cognitive detox
        elif overall >= 55:
            level = 4

        # Level 3: Adaptive slowdown
        elif overall >= 40:
            level = 3

        # Level 2: Behavioral nudge
        elif overall >= 25 or watch_min > 20:
            level = 2

        # Level 1: Gentle awareness
        elif overall >= 15 or watch_min > 10:
            level = 1

        if level == 0:
            return {
                'level': 0,
                'name':  'Safe',
                'color': '#22c55e',
                'icon':  '✅',
                'message': 'You are scrolling mindfully. Keep it up!',
                'action': 'none',
                'lock_duration': 0,
                'tips': [],
                'timestamp': datetime.now().isoformat()
            }

        action = dict(self.INTERVENTIONS[level])
        action['timestamp']    = datetime.now().isoformat()
        action['trigger_score'] = overall
        action['should_trigger'] = level >= 2

        return action

    def get_recommendations(self, scores_history: list) -> list:
        """Generate AI recommendations from score history."""
        if not scores_history:
            return [
                "Start your session to receive personalized recommendations.",
            ]

        recs = []
        avg_risk    = sum(s.get('overall_risk', 0) for s in scores_history) / len(scores_history)
        avg_blink   = sum(s.get('blink_rate', 15) for s in scores_history) / len(scores_history)
        night_use   = any(s.get('night_penalty', 0) > 30 for s in scores_history)
        long_session= any(s.get('watch_time_min', 0) > 45 for s in scores_history)

        if avg_risk > 60:
            recs.append("⚠️ Your average risk score is high. Consider setting daily reel limits.")
        if avg_blink < 10:
            recs.append("👁️ Your blink rate is low. Take regular eye breaks using the 20-20-20 rule.")
        if night_use:
            recs.append("🌙 You scroll late at night. Try avoiding reels after 10pm for better sleep.")
        if long_session:
            recs.append("⏱️ Your sessions often exceed 45 minutes. Set a 30-minute daily reel limit.")
        if not recs:
            recs.append("✅ Great scrolling habits! Keep maintaining mindful usage patterns.")

        return recs
