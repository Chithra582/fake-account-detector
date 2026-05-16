"""
Flask API Backend for Fake Social Media Account Detection
Serves predictions and analytics to the web dashboard
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import numpy as np
import os

app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)

# Load model and metadata
model = joblib.load("models/best_model.pkl")
features = joblib.load("models/features.pkl")
with open("models/model_meta.json") as f:
    model_meta = json.load(f)

FEATURE_GROUPS = model_meta["feature_groups"]
BEST_MODEL = model_meta["best_model"]
ALL_RESULTS = model_meta["all_results"]

RISK_THRESHOLDS = {
    "low": 0.35,
    "medium": 0.60,
    "high": 0.80
}

def compute_risk_level(probability):
    if probability < RISK_THRESHOLDS["low"]:
        return "Genuine", "low", "#10b981"
    elif probability < RISK_THRESHOLDS["medium"]:
        return "Suspicious", "medium", "#f59e0b"
    elif probability < RISK_THRESHOLDS["high"]:
        return "Likely Fake", "high", "#f97316"
    else:
        return "Fake Account", "critical", "#ef4444"

def compute_feature_scores(input_data):
    """Generate per-group risk scores for the input profile."""
    scores = {}
    for group, group_features in FEATURE_GROUPS.items():
        group_vals = {f: input_data.get(f, 0) for f in group_features if f in input_data}
        if not group_vals:
            continue
        # Simple heuristic scoring (normalize and weight)
        risk_score = 0
        count = 0
        for feat, val in group_vals.items():
            if feat == "has_profile_picture":
                risk_score += (1 - val) * 30
            elif feat == "account_age_days":
                risk_score += max(0, (365 - val) / 365) * 25
            elif feat == "posts_per_day":
                risk_score += min(val / 10, 1) * 40
            elif feat == "follower_following_ratio":
                r = val
                risk_score += (1 - min(r / 2, 1)) * 20 if r < 2 else min(r / 10, 1) * 20
            elif feat == "fake_follower_pct":
                risk_score += val * 50
            elif feat == "uses_automation":
                risk_score += val * 40
            elif feat == "spam_keyword_count":
                risk_score += min(val / 5, 1) * 35
            elif feat == "engagement_rate":
                risk_score += (1 - min(val / 0.1, 1)) * 20
            elif feat == "profile_completeness":
                risk_score += (1 - val) * 25
            else:
                risk_score += 0
            count += 1
        scores[group] = round(min(risk_score / max(count, 1), 100), 1)
    return scores


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Build feature vector
    feature_vector = np.array([[data.get(f, 0) for f in features]])

    # Predict
    probability = float(model.predict_proba(feature_vector)[0][1])
    prediction = int(model.predict(feature_vector)[0])
    verdict, risk_level, color = compute_risk_level(probability)
    group_scores = compute_feature_scores(data)

    # Top risk factors
    feature_vals = {f: data.get(f, 0) for f in features}

    return jsonify({
        "prediction": prediction,
        "probability": round(probability, 4),
        "verdict": verdict,
        "risk_level": risk_level,
        "risk_color": color,
        "confidence": round(abs(probability - 0.5) * 2, 4),
        "group_risk_scores": group_scores,
        "model_used": BEST_MODEL
    })


@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    data = request.json
    if not data or 'profiles' not in data:
        return jsonify({"error": "No profiles provided"}), 400

    results = []
    for profile in data['profiles']:
        feature_vector = np.array([[profile.get(f, 0) for f in features]])
        probability = float(model.predict_proba(feature_vector)[0][1])
        verdict, risk_level, color = compute_risk_level(probability)
        results.append({
            "id": profile.get("id", "unknown"),
            "username": profile.get("username", "anonymous"),
            "probability": round(probability, 4),
            "verdict": verdict,
            "risk_level": risk_level,
            "risk_color": color
        })

    return jsonify({
        "results": results,
        "total": len(results),
        "flagged": sum(1 for r in results if r['risk_level'] in ['high', 'critical']),
        "suspicious": sum(1 for r in results if r['risk_level'] == 'medium')
    })


@app.route('/api/model_stats', methods=['GET'])
def model_stats():
    return jsonify({
        "best_model": BEST_MODEL,
        "models": ALL_RESULTS,
        "feature_count": model_meta["n_features"],
        "feature_groups": list(FEATURE_GROUPS.keys()),
        "train_samples": model_meta["train_samples"],
        "test_samples": model_meta["test_samples"]
    })


@app.route('/api/demo_profiles', methods=['GET'])
def demo_profiles():
    """Return sample profiles for demo purposes."""
    demo = [
        {
            "name": "Sarah Johnson", "username": "sarahjohnson_real",
            "type": "Real User", "expected": "Genuine",
            "has_profile_picture": 1, "username_digit_ratio": 0.05, "username_length": 15,
            "has_bio": 1, "bio_length": 120, "bio_url_count": 1, "has_external_url": 1, "is_verified": 0,
            "account_age_days": 1200, "posts_count": 340, "posts_per_day": 0.28,
            "avg_post_likes": 85, "avg_post_comments": 12, "avg_post_shares": 5,
            "story_frequency": 0.4, "live_frequency": 0.05,
            "followers_count": 850, "following_count": 620, "follower_following_ratio": 1.37,
            "mutual_friends_ratio": 0.65, "fake_follower_pct": 0.04,
            "avg_daily_posts": 0.28, "posting_hour_variance": 8.0, "night_posting_ratio": 0.15,
            "weekend_posting_ratio": 0.4, "hashtag_avg": 4, "mention_avg": 2,
            "caption_length_avg": 95, "emoji_usage_ratio": 0.35, "uses_automation": 0,
            "content_diversity_score": 0.75, "repost_ratio": 0.15, "original_content_ratio": 0.82,
            "spam_keyword_count": 0, "link_in_posts_ratio": 0.08, "sentiment_variance": 0.55,
            "engagement_rate": 0.07, "comment_to_like_ratio": 0.14, "profile_completeness": 0.90,
            "response_time_hours": 6.0
        },
        {
            "name": "BOT_99821", "username": "user98238932",
            "type": "Bot Account", "expected": "Fake Account",
            "has_profile_picture": 0, "username_digit_ratio": 0.78, "username_length": 14,
            "has_bio": 0, "bio_length": 0, "bio_url_count": 0, "has_external_url": 0, "is_verified": 0,
            "account_age_days": 45, "posts_count": 820, "posts_per_day": 18.2,
            "avg_post_likes": 3, "avg_post_comments": 1, "avg_post_shares": 0,
            "story_frequency": 0.02, "live_frequency": 0.0,
            "followers_count": 42, "following_count": 4980, "follower_following_ratio": 0.008,
            "mutual_friends_ratio": 0.02, "fake_follower_pct": 0.91,
            "avg_daily_posts": 55.0, "posting_hour_variance": 0.5, "night_posting_ratio": 0.75,
            "weekend_posting_ratio": 0.52, "hashtag_avg": 28, "mention_avg": 18,
            "caption_length_avg": 8, "emoji_usage_ratio": 0.05, "uses_automation": 1,
            "content_diversity_score": 0.08, "repost_ratio": 0.95, "original_content_ratio": 0.05,
            "spam_keyword_count": 4, "link_in_posts_ratio": 0.88, "sentiment_variance": 0.05,
            "engagement_rate": 0.002, "comment_to_like_ratio": 0.02, "profile_completeness": 0.10,
            "response_time_hours": 0.2
        },
        {
            "name": "Mark_Zuckerberg_Fan", "username": "markzuck_real2024",
            "type": "Impersonator", "expected": "Likely Fake",
            "has_profile_picture": 1, "username_digit_ratio": 0.24, "username_length": 17,
            "has_bio": 1, "bio_length": 75, "bio_url_count": 2, "has_external_url": 1, "is_verified": 0,
            "account_age_days": 95, "posts_count": 180, "posts_per_day": 1.89,
            "avg_post_likes": 55, "avg_post_comments": 9, "avg_post_shares": 8,
            "story_frequency": 0.2, "live_frequency": 0.05,
            "followers_count": 2850, "following_count": 3500, "follower_following_ratio": 0.81,
            "mutual_friends_ratio": 0.12, "fake_follower_pct": 0.78,
            "avg_daily_posts": 5.0, "posting_hour_variance": 4.0, "night_posting_ratio": 0.38,
            "weekend_posting_ratio": 0.5, "hashtag_avg": 15, "mention_avg": 12,
            "caption_length_avg": 55, "emoji_usage_ratio": 0.20, "uses_automation": 1,
            "content_diversity_score": 0.22, "repost_ratio": 0.72, "original_content_ratio": 0.25,
            "spam_keyword_count": 3, "link_in_posts_ratio": 0.65, "sentiment_variance": 0.20,
            "engagement_rate": 0.015, "comment_to_like_ratio": 0.05, "profile_completeness": 0.55,
            "response_time_hours": 2.5
        },
        {
            "name": "TechBlogger Alex", "username": "alexcodes",
            "type": "Real Influencer", "expected": "Genuine",
            "has_profile_picture": 1, "username_digit_ratio": 0.0, "username_length": 9,
            "has_bio": 1, "bio_length": 155, "bio_url_count": 1, "has_external_url": 1, "is_verified": 1,
            "account_age_days": 2200, "posts_count": 980, "posts_per_day": 0.45,
            "avg_post_likes": 2200, "avg_post_comments": 185, "avg_post_shares": 42,
            "story_frequency": 0.7, "live_frequency": 0.15,
            "followers_count": 48500, "following_count": 1200, "follower_following_ratio": 40.4,
            "mutual_friends_ratio": 0.45, "fake_follower_pct": 0.08,
            "avg_daily_posts": 0.45, "posting_hour_variance": 12.0, "night_posting_ratio": 0.18,
            "weekend_posting_ratio": 0.35, "hashtag_avg": 6, "mention_avg": 3,
            "caption_length_avg": 175, "emoji_usage_ratio": 0.42, "uses_automation": 0,
            "content_diversity_score": 0.85, "repost_ratio": 0.10, "original_content_ratio": 0.90,
            "spam_keyword_count": 0, "link_in_posts_ratio": 0.12, "sentiment_variance": 0.68,
            "engagement_rate": 0.09, "comment_to_like_ratio": 0.12, "profile_completeness": 0.98,
            "response_time_hours": 18.0
        }
    ]
    return jsonify(demo)


@app.route('/api/features', methods=['GET'])
def get_features():
    return jsonify({
        "features": features,
        "feature_groups": FEATURE_GROUPS
    })


@app.route('/')
def index():
    return app.send_static_file('index.html')


if __name__ == '__main__':
    print("\n[ShieldAI] Fake Account Detection API running at http://localhost:5000")
    app.run(host="0.0.0.0", port=7860)
