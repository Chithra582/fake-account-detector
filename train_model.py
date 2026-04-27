"""
ML Training Pipeline for Fake Social Media Account Detection
Trains multiple classifiers and selects the best model
"""

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score, accuracy_score,
    average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.over_sampling import SMOTE
import xgboost as xgb

warnings.filterwarnings('ignore')

MODELS_DIR = "models"
PLOTS_DIR = "static/plots"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

FEATURE_GROUPS = {
    "Identity": ["has_profile_picture", "username_digit_ratio", "username_length",
                 "has_bio", "bio_length", "bio_url_count", "has_external_url",
                 "is_verified", "profile_completeness"],
    "Activity": ["account_age_days", "posts_count", "posts_per_day",
                 "avg_post_likes", "avg_post_comments", "avg_post_shares",
                 "story_frequency", "live_frequency"],
    "Network": ["followers_count", "following_count", "follower_following_ratio",
                "mutual_friends_ratio", "fake_follower_pct"],
    "Behavioral": ["avg_daily_posts", "posting_hour_variance", "night_posting_ratio",
                   "weekend_posting_ratio", "hashtag_avg", "mention_avg",
                   "caption_length_avg", "emoji_usage_ratio", "uses_automation"],
    "Content": ["content_diversity_score", "repost_ratio", "original_content_ratio",
                "spam_keyword_count", "link_in_posts_ratio", "sentiment_variance"],
    "Engagement": ["engagement_rate", "comment_to_like_ratio", "response_time_hours"]
}

def load_data():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    with open("data/features.json") as f:
        features = json.load(f)
    X_train = train[features].values
    y_train = train['label'].values
    X_test = test[features].values
    y_test = test['label'].values
    return X_train, y_train, X_test, y_test, features


def build_models():
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_split=5,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=1.5, random_state=42,
            eval_metric='logloss'
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42
        ),
        "Logistic Regression": Pipeline([
            ('scaler', RobustScaler()),
            ('clf', LogisticRegression(
                C=1.0, class_weight='balanced',
                max_iter=1000, random_state=42
            ))
        ]),
        "SVM": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(
                C=1.0, kernel='rbf', probability=True,
                class_weight='balanced', random_state=42
            ))
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=15, min_samples_split=10,
            class_weight='balanced', random_state=42
        )
    }


def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'], ax=ax)
    ax.set_title(f'{model_name}\nConfusion Matrix', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    fname = f"{PLOTS_DIR}/cm_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    return fname


def plot_roc_curves(results):
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4']
    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(res['y_test'], res['y_prob'])
        auc = res['roc_auc']
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Random Classifier')
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — All Models', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/roc_curves.png", dpi=130, bbox_inches='tight')
    plt.close()


def plot_feature_importance(model, features, model_name):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'named_steps'):
        clf = model.named_steps.get('clf')
        if hasattr(clf, 'coef_'):
            importances = np.abs(clf.coef_[0])
        else:
            return None
    else:
        return None

    idx = np.argsort(importances)[::-1][:20]
    top_features = [features[i] for i in idx]
    top_imp = importances[idx]

    # Color by feature group
    group_colors = {
        "Identity": "#6366f1", "Activity": "#10b981", "Network": "#f59e0b",
        "Behavioral": "#ef4444", "Content": "#8b5cf6", "Engagement": "#06b6d4"
    }
    feature_to_group = {}
    for group, feats in FEATURE_GROUPS.items():
        for f in feats:
            feature_to_group[f] = group
    bar_colors = [group_colors.get(feature_to_group.get(f, ""), '#aaa') for f in top_features]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(top_features)), top_imp, color=bar_colors, edgecolor='white', height=0.7)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance Score', fontsize=11)
    ax.set_title(f'{model_name}\nTop 20 Feature Importances', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=g) for g, c in group_colors.items()]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    plt.tight_layout()
    fname = f"{PLOTS_DIR}/feature_importance_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    return fname


def plot_model_comparison(results):
    models = list(results.keys())
    metrics = ['accuracy', 'f1_score', 'roc_auc', 'precision', 'recall']
    labels = ['Accuracy', 'F1 Score', 'ROC-AUC', 'Precision', 'Recall']

    x = np.arange(len(models))
    width = 0.15
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']

    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = [results[m][metric] for m in models]
        bars = ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85, edgecolor='white')

    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(models, fontsize=10, rotation=15)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4, lw=1)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/model_comparison.png", dpi=130, bbox_inches='tight')
    plt.close()


def plot_data_distribution(train_df):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    key_features = [
        ('followers_count', 'Follower Count'),
        ('posts_per_day', 'Posts Per Day'),
        ('follower_following_ratio', 'Follower/Following Ratio'),
        ('avg_daily_posts', 'Avg Daily Posts'),
        ('engagement_rate', 'Engagement Rate'),
        ('fake_follower_pct', 'Fake Follower %')
    ]
    colors = {'Real': '#10b981', 'Fake': '#ef4444'}

    for ax, (feat, title) in zip(axes.flat, key_features):
        for label, name in [(0, 'Real'), (1, 'Fake')]:
            subset = train_df[train_df['label'] == label][feat]
            clipped = subset.clip(subset.quantile(0.01), subset.quantile(0.99))
            ax.hist(clipped, bins=35, alpha=0.65, color=colors[name], label=name, density=True)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel(feat, fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    fig.suptitle('Feature Distributions: Real vs Fake Accounts', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/data_distribution.png", dpi=130, bbox_inches='tight')
    plt.close()


def plot_correlation_heatmap(train_df, features):
    top_corr_features = (
        train_df[features].corrwith(train_df['label'])
        .abs().sort_values(ascending=False).head(20).index.tolist()
    )
    corr = train_df[top_corr_features].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, mask=mask, cmap='RdYlGn', center=0, annot=True,
                fmt='.2f', linewidths=0.4, ax=ax, annot_kws={"size": 7},
                square=True, cbar_kws={'shrink': 0.7})
    ax.set_title('Feature Correlation Heatmap (Top 20)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/correlation_heatmap.png", dpi=120, bbox_inches='tight')
    plt.close()


def train_and_evaluate():
    print("=" * 60)
    print("FAKE SOCIAL MEDIA ACCOUNT DETECTION — ML PIPELINE")
    print("=" * 60)

    print("\n[1/6] Loading data...")
    X_train, y_train, X_test, y_test, features = load_data()
    train_df = pd.read_csv("data/train.csv")

    print(f"     Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    print(f"     Features: {len(features)}")
    print(f"     Train Class Balance: Real={sum(y_train==0)} | Fake={sum(y_train==1)}")

    print("\n[2/6] Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"     After SMOTE: {X_train_bal.shape[0]} samples (balanced)")

    print("\n[3/6] Generating EDA plots...")
    plot_data_distribution(train_df)
    plot_correlation_heatmap(train_df, features)
    print("     Saved distribution & correlation plots")

    print("\n[4/6] Training models...")
    models = build_models()
    results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train_bal, y_train_bal)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        report = classification_report(y_test, y_pred, output_dict=True)

        cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc', n_jobs=-1)

        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "precision": report['1']['precision'],
            "recall": report['1']['recall'],
            "cv_auc_mean": cv_scores.mean(),
            "cv_auc_std": cv_scores.std(),
            "y_test": y_test,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "report": report,
            "model": model
        }

        print(f"    Accuracy: {results[name]['accuracy']:.4f} | F1: {results[name]['f1_score']:.4f} | AUC: {results[name]['roc_auc']:.4f} | CV-AUC: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
        plot_confusion_matrix(y_test, y_pred, name)

    print("\n[5/6] Generating performance plots...")
    plot_roc_curves(results)
    plot_model_comparison(results)

    best_model_name = max(results, key=lambda k: results[k]['roc_auc'])
    best_model = results[best_model_name]['model']
    plot_feature_importance(best_model, features, best_model_name)
    print(f"     Best model: {best_model_name} (AUC={results[best_model_name]['roc_auc']:.4f})")

    print("\n[6/6] Saving best model & metadata...")
    joblib.dump(best_model, f"{MODELS_DIR}/best_model.pkl")
    joblib.dump(features, f"{MODELS_DIR}/features.pkl")

    model_meta = {
        "best_model": best_model_name,
        "all_results": {
            name: {k: v for k, v in res.items() if k not in ['y_test', 'y_pred', 'y_prob', 'model', 'report']}
            for name, res in results.items()
        },
        "feature_groups": FEATURE_GROUPS,
        "n_features": len(features),
        "test_samples": len(y_test),
        "train_samples": len(y_train)
    }
    with open(f"{MODELS_DIR}/model_meta.json", "w") as f:
        json.dump(model_meta, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  BEST MODEL: {best_model_name}")
    print(f"  ROC-AUC:    {results[best_model_name]['roc_auc']:.4f}")
    print(f"  Accuracy:   {results[best_model_name]['accuracy']:.4f}")
    print(f"  F1-Score:   {results[best_model_name]['f1_score']:.4f}")
    print("=" * 60)

    return best_model, features, results


if __name__ == "__main__":
    train_and_evaluate()
