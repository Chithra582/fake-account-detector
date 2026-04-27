"""
Synthetic Dataset Generator for Fake Social Media Account Detection
Generates realistic fake and real social media profile data with behavioral features
"""

import numpy as np
import pandas as pd
import os
import json

np.random.seed(42)

def generate_real_profiles(n=2500):
    """Generate realistic profiles for genuine users."""
    profiles = []
    for i in range(n):
        account_age_days = np.random.randint(180, 3650)
        followers = int(np.random.lognormal(mean=5.5, sigma=1.8))
        following = int(np.random.lognormal(mean=5.0, sigma=1.5))
        posts = int(np.random.lognormal(mean=4.5, sigma=1.3))

        profile = {
            # Identity features
            "has_profile_picture": np.random.choice([1, 0], p=[0.94, 0.06]),
            "username_digit_ratio": np.random.beta(1.5, 10),
            "username_length": np.random.randint(5, 22),
            "has_bio": np.random.choice([1, 0], p=[0.82, 0.18]),
            "bio_length": np.random.randint(10, 160) if np.random.random() > 0.18 else 0,
            "bio_url_count": np.random.choice([0, 1, 2], p=[0.55, 0.40, 0.05]),
            "has_external_url": np.random.choice([1, 0], p=[0.45, 0.55]),
            "is_verified": np.random.choice([1, 0], p=[0.03, 0.97]),

            # Activity features
            "account_age_days": account_age_days,
            "posts_count": posts,
            "posts_per_day": round(posts / max(account_age_days, 1), 4),
            "avg_post_likes": int(np.random.lognormal(mean=3.8, sigma=1.2)),
            "avg_post_comments": int(np.random.lognormal(mean=2.2, sigma=1.0)),
            "avg_post_shares": int(np.random.lognormal(mean=1.5, sigma=1.1)),
            "story_frequency": np.random.beta(2, 5),
            "live_frequency": np.random.beta(1, 8),

            # Network features
            "followers_count": followers,
            "following_count": following,
            "follower_following_ratio": round(followers / max(following, 1), 4),
            "mutual_friends_ratio": np.random.beta(5, 3),
            "fake_follower_pct": np.random.beta(1.5, 10),

            # Behavioral features
            "avg_daily_posts": np.random.uniform(0.1, 4.0),
            "posting_hour_variance": np.random.uniform(5.0, 23.0),
            "night_posting_ratio": np.random.beta(2, 8),
            "weekend_posting_ratio": np.random.beta(3, 5),
            "hashtag_avg": np.random.randint(0, 12),
            "mention_avg": np.random.randint(0, 8),
            "caption_length_avg": np.random.randint(20, 220),
            "emoji_usage_ratio": np.random.beta(3, 5),
            "uses_automation": np.random.choice([1, 0], p=[0.05, 0.95]),

            # Content features
            "content_diversity_score": np.random.beta(6, 3),
            "repost_ratio": np.random.beta(2, 8),
            "original_content_ratio": np.random.beta(7, 3),
            "spam_keyword_count": np.random.choice([0, 1], p=[0.92, 0.08]),
            "link_in_posts_ratio": np.random.beta(2, 9),
            "sentiment_variance": np.random.uniform(0.2, 0.9),

            # Engagement quality
            "engagement_rate": np.random.beta(3, 10),
            "comment_to_like_ratio": np.random.beta(2, 8),
            "profile_completeness": np.random.beta(7, 2),
            "response_time_hours": np.random.exponential(scale=12),

            "label": 0  # Real
        }
        profiles.append(profile)
    return profiles


def generate_fake_profiles(n=1500):
    """Generate profiles for fake/bot accounts."""
    profiles = []
    fake_types = ["bot", "spam", "impersonator", "purchased_followers"]
    weights = [0.35, 0.30, 0.20, 0.15]

    for i in range(n):
        fake_type = np.random.choice(fake_types, p=weights)
        account_age_days = np.random.randint(1, 365)

        if fake_type == "bot":
            followers = int(np.random.lognormal(mean=3.5, sigma=1.2))
            following = int(np.random.lognormal(mean=7.0, sigma=0.8))
            posts = int(np.random.lognormal(mean=6.5, sigma=1.0))
            profile = {
                "has_profile_picture": np.random.choice([1, 0], p=[0.35, 0.65]),
                "username_digit_ratio": np.random.beta(8, 3),
                "username_length": np.random.randint(8, 30),
                "has_bio": np.random.choice([1, 0], p=[0.25, 0.75]),
                "bio_length": np.random.randint(0, 40) if np.random.random() > 0.75 else 0,
                "bio_url_count": np.random.choice([0, 1, 2, 3], p=[0.40, 0.35, 0.15, 0.10]),
                "has_external_url": np.random.choice([1, 0], p=[0.55, 0.45]),
                "is_verified": 0,
                "account_age_days": account_age_days,
                "posts_count": posts,
                "posts_per_day": round(posts / max(account_age_days, 1), 4),
                "avg_post_likes": int(np.random.lognormal(mean=1.5, sigma=0.8)),
                "avg_post_comments": int(np.random.lognormal(mean=0.5, sigma=0.7)),
                "avg_post_shares": int(np.random.lognormal(mean=0.3, sigma=0.5)),
                "story_frequency": np.random.beta(1, 10),
                "live_frequency": np.random.beta(1, 20),
                "followers_count": followers,
                "following_count": following,
                "follower_following_ratio": round(followers / max(following, 1), 4),
                "mutual_friends_ratio": np.random.beta(1, 9),
                "fake_follower_pct": np.random.beta(8, 3),
                "avg_daily_posts": np.random.uniform(10, 80),
                "posting_hour_variance": np.random.uniform(0.1, 4.0),
                "night_posting_ratio": np.random.beta(5, 3),
                "weekend_posting_ratio": np.random.beta(5, 5),
                "hashtag_avg": np.random.randint(15, 35),
                "mention_avg": np.random.randint(8, 25),
                "caption_length_avg": np.random.randint(0, 50),
                "emoji_usage_ratio": np.random.beta(1, 8),
                "uses_automation": np.random.choice([1, 0], p=[0.88, 0.12]),
                "content_diversity_score": np.random.beta(2, 8),
                "repost_ratio": np.random.beta(8, 2),
                "original_content_ratio": np.random.beta(2, 8),
                "spam_keyword_count": np.random.choice([0, 1, 2, 3, 4], p=[0.20, 0.30, 0.25, 0.15, 0.10]),
                "link_in_posts_ratio": np.random.beta(7, 3),
                "sentiment_variance": np.random.uniform(0.01, 0.2),
                "engagement_rate": np.random.beta(1, 12),
                "comment_to_like_ratio": np.random.beta(1, 12),
                "profile_completeness": np.random.beta(2, 8),
                "response_time_hours": np.random.exponential(scale=0.5),
                "label": 1
            }

        elif fake_type == "spam":
            followers = int(np.random.lognormal(mean=4.0, sigma=1.5))
            following = int(np.random.lognormal(mean=7.5, sigma=0.6))
            posts = int(np.random.lognormal(mean=5.5, sigma=0.9))
            profile = {
                "has_profile_picture": np.random.choice([1, 0], p=[0.55, 0.45]),
                "username_digit_ratio": np.random.beta(6, 4),
                "username_length": np.random.randint(10, 28),
                "has_bio": np.random.choice([1, 0], p=[0.45, 0.55]),
                "bio_length": np.random.randint(0, 80) if np.random.random() > 0.55 else 0,
                "bio_url_count": np.random.choice([0, 1, 2, 3], p=[0.25, 0.35, 0.25, 0.15]),
                "has_external_url": np.random.choice([1, 0], p=[0.70, 0.30]),
                "is_verified": 0,
                "account_age_days": account_age_days,
                "posts_count": posts,
                "posts_per_day": round(posts / max(account_age_days, 1), 4),
                "avg_post_likes": int(np.random.lognormal(mean=2.0, sigma=0.9)),
                "avg_post_comments": int(np.random.lognormal(mean=0.8, sigma=0.7)),
                "avg_post_shares": int(np.random.lognormal(mean=0.5, sigma=0.6)),
                "story_frequency": np.random.beta(1, 12),
                "live_frequency": np.random.beta(1, 25),
                "followers_count": followers,
                "following_count": following,
                "follower_following_ratio": round(followers / max(following, 1), 4),
                "mutual_friends_ratio": np.random.beta(1.5, 8),
                "fake_follower_pct": np.random.beta(7, 3),
                "avg_daily_posts": np.random.uniform(8, 50),
                "posting_hour_variance": np.random.uniform(0.5, 5.0),
                "night_posting_ratio": np.random.beta(4, 4),
                "weekend_posting_ratio": np.random.beta(5, 5),
                "hashtag_avg": np.random.randint(12, 30),
                "mention_avg": np.random.randint(5, 20),
                "caption_length_avg": np.random.randint(5, 80),
                "emoji_usage_ratio": np.random.beta(2, 7),
                "uses_automation": np.random.choice([1, 0], p=[0.75, 0.25]),
                "content_diversity_score": np.random.beta(2, 7),
                "repost_ratio": np.random.beta(7, 3),
                "original_content_ratio": np.random.beta(2, 7),
                "spam_keyword_count": np.random.choice([0, 1, 2, 3, 4, 5], p=[0.10, 0.20, 0.25, 0.20, 0.15, 0.10]),
                "link_in_posts_ratio": np.random.beta(8, 2),
                "sentiment_variance": np.random.uniform(0.05, 0.3),
                "engagement_rate": np.random.beta(1.5, 11),
                "comment_to_like_ratio": np.random.beta(1.5, 10),
                "profile_completeness": np.random.beta(3, 7),
                "response_time_hours": np.random.exponential(scale=1.0),
                "label": 1
            }

        elif fake_type == "impersonator":
            followers = int(np.random.lognormal(mean=5.0, sigma=1.5))
            following = int(np.random.lognormal(mean=6.0, sigma=1.2))
            posts = int(np.random.lognormal(mean=4.0, sigma=1.1))
            profile = {
                "has_profile_picture": np.random.choice([1, 0], p=[0.85, 0.15]),
                "username_digit_ratio": np.random.beta(5, 4),
                "username_length": np.random.randint(8, 25),
                "has_bio": np.random.choice([1, 0], p=[0.75, 0.25]),
                "bio_length": np.random.randint(20, 130) if np.random.random() > 0.25 else 0,
                "bio_url_count": np.random.choice([0, 1, 2], p=[0.35, 0.45, 0.20]),
                "has_external_url": np.random.choice([1, 0], p=[0.60, 0.40]),
                "is_verified": 0,
                "account_age_days": account_age_days,
                "posts_count": posts,
                "posts_per_day": round(posts / max(account_age_days, 1), 4),
                "avg_post_likes": int(np.random.lognormal(mean=2.8, sigma=1.1)),
                "avg_post_comments": int(np.random.lognormal(mean=1.5, sigma=0.9)),
                "avg_post_shares": int(np.random.lognormal(mean=1.2, sigma=0.9)),
                "story_frequency": np.random.beta(2, 8),
                "live_frequency": np.random.beta(1, 15),
                "followers_count": followers,
                "following_count": following,
                "follower_following_ratio": round(followers / max(following, 1), 4),
                "mutual_friends_ratio": np.random.beta(2, 8),
                "fake_follower_pct": np.random.beta(6, 4),
                "avg_daily_posts": np.random.uniform(2, 15),
                "posting_hour_variance": np.random.uniform(2.0, 12.0),
                "night_posting_ratio": np.random.beta(3, 5),
                "weekend_posting_ratio": np.random.beta(4, 5),
                "hashtag_avg": np.random.randint(8, 22),
                "mention_avg": np.random.randint(4, 18),
                "caption_length_avg": np.random.randint(10, 120),
                "emoji_usage_ratio": np.random.beta(2, 6),
                "uses_automation": np.random.choice([1, 0], p=[0.55, 0.45]),
                "content_diversity_score": np.random.beta(3, 7),
                "repost_ratio": np.random.beta(6, 4),
                "original_content_ratio": np.random.beta(3, 7),
                "spam_keyword_count": np.random.choice([0, 1, 2, 3], p=[0.30, 0.30, 0.25, 0.15]),
                "link_in_posts_ratio": np.random.beta(6, 4),
                "sentiment_variance": np.random.uniform(0.1, 0.5),
                "engagement_rate": np.random.beta(2, 10),
                "comment_to_like_ratio": np.random.beta(2, 9),
                "profile_completeness": np.random.beta(5, 5),
                "response_time_hours": np.random.exponential(scale=5.0),
                "label": 1
            }

        else:  # purchased_followers
            followers = int(np.random.lognormal(mean=8.0, sigma=1.0))
            following = int(np.random.lognormal(mean=5.5, sigma=1.3))
            posts = int(np.random.lognormal(mean=3.8, sigma=1.1))
            profile = {
                "has_profile_picture": np.random.choice([1, 0], p=[0.88, 0.12]),
                "username_digit_ratio": np.random.beta(3, 6),
                "username_length": np.random.randint(5, 20),
                "has_bio": np.random.choice([1, 0], p=[0.70, 0.30]),
                "bio_length": np.random.randint(10, 150) if np.random.random() > 0.30 else 0,
                "bio_url_count": np.random.choice([0, 1, 2], p=[0.50, 0.40, 0.10]),
                "has_external_url": np.random.choice([1, 0], p=[0.50, 0.50]),
                "is_verified": 0,
                "account_age_days": account_age_days,
                "posts_count": posts,
                "posts_per_day": round(posts / max(account_age_days, 1), 4),
                "avg_post_likes": int(np.random.lognormal(mean=4.5, sigma=1.5)),
                "avg_post_comments": int(np.random.lognormal(mean=1.0, sigma=0.8)),
                "avg_post_shares": int(np.random.lognormal(mean=0.8, sigma=0.8)),
                "story_frequency": np.random.beta(2, 7),
                "live_frequency": np.random.beta(1, 12),
                "followers_count": followers,
                "following_count": following,
                "follower_following_ratio": round(followers / max(following, 1), 4),
                "mutual_friends_ratio": np.random.beta(2, 8),
                "fake_follower_pct": np.random.beta(9, 2),
                "avg_daily_posts": np.random.uniform(0.2, 5.0),
                "posting_hour_variance": np.random.uniform(3.0, 20.0),
                "night_posting_ratio": np.random.beta(2, 7),
                "weekend_posting_ratio": np.random.beta(3, 6),
                "hashtag_avg": np.random.randint(5, 18),
                "mention_avg": np.random.randint(0, 10),
                "caption_length_avg": np.random.randint(10, 180),
                "emoji_usage_ratio": np.random.beta(3, 6),
                "uses_automation": np.random.choice([1, 0], p=[0.40, 0.60]),
                "content_diversity_score": np.random.beta(4, 6),
                "repost_ratio": np.random.beta(4, 6),
                "original_content_ratio": np.random.beta(5, 5),
                "spam_keyword_count": np.random.choice([0, 1, 2], p=[0.55, 0.30, 0.15]),
                "link_in_posts_ratio": np.random.beta(4, 6),
                "sentiment_variance": np.random.uniform(0.15, 0.7),
                "engagement_rate": np.random.beta(3, 7),
                "comment_to_like_ratio": np.random.beta(1, 12),
                "profile_completeness": np.random.beta(5, 4),
                "response_time_hours": np.random.exponential(scale=8.0),
                "label": 1
            }

        profiles.append(profile)
    return profiles


def add_noise(df, noise_level=0.03):
    """Add slight label noise to make data more realistic."""
    flip_idx = np.random.choice(df.index, size=int(len(df) * noise_level), replace=False)
    df.loc[flip_idx, 'label'] = 1 - df.loc[flip_idx, 'label']
    return df


def generate_dataset():
    os.makedirs("data", exist_ok=True)
    print("Generating real profiles...")
    real = generate_real_profiles(2500)
    print("Generating fake profiles...")
    fake = generate_fake_profiles(1500)

    all_profiles = real + fake
    df = pd.DataFrame(all_profiles)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = add_noise(df, noise_level=0.02)

    # Clip unrealistic values
    df['posts_per_day'] = df['posts_per_day'].clip(0, 100)
    df['follower_following_ratio'] = df['follower_following_ratio'].clip(0, 500)
    df['avg_daily_posts'] = df['avg_daily_posts'].clip(0, 150)
    df['response_time_hours'] = df['response_time_hours'].clip(0, 200)

    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    df.to_csv("data/social_media_profiles.csv", index=False)
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    print(f"\nDataset generated successfully!")
    print(f"Total profiles: {len(df)}")
    print(f"Real accounts: {(df['label'] == 0).sum()} ({(df['label'] == 0).mean()*100:.1f}%)")
    print(f"Fake accounts: {(df['label'] == 1).sum()} ({(df['label'] == 1).mean()*100:.1f}%)")
    print(f"Train set: {len(train_df)} | Test set: {len(test_df)}")
    print(f"Features: {len(df.columns) - 1}")
    print("\nSaved to data/")

    # Save feature list
    features = [c for c in df.columns if c != 'label']
    with open("data/features.json", "w") as f:
        json.dump(features, f, indent=2)

    return df


if __name__ == "__main__":
    generate_dataset()
