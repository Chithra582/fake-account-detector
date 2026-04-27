# Fake Social Media Account Detection 🛡️

An end-to-end Machine Learning system designed to detect fake, bot, and spam accounts on social media platforms. The project includes a synthetic data generator, a comprehensive training pipeline, and a responsive web dashboard for real-time profile analysis.

## Key Features

- **Synthetic Data Generation**: Creates realistic profiles for genuine users, bots, spammers, and impersonators.
- **Ensemble ML Pipeline**: Trains and compares multiple models (Random Forest, XGBoost, Gradient Boosting, SVM, etc.).
- **Feature Importance Analysis**: Identifies which behavioral patterns (e.g., posting frequency, hashtag usage) most indicate a fake account.
- **Web Dashboard**: Interactive Flask-based dashboard for profile auditing and model performance visualization.
- **Automated Setup**: One-click script to go from zero to a running application.

---

## Project Structure

```text
├── app.py                # Flask API and Web Server
├── generate_dataset.py   # Synthetic data generation script
├── train_model.py        # ML training and evaluation pipeline
├── setup_and_run.py      # Automated setup & launch script
├── requirements.txt      # Python dependencies
├── data/                 # Generated datasets (CSV)
├── models/               # Saved models and metadata (joblib/json)
└── static/               # Dashboard frontend (HTML/CSS/JS)
    └── plots/            # Generated performance visualizations
```

---

## Quick Start

### 1. Prerequisites
- **Python 3.8+**
- **pip** (Python package manager)

### 2. Setup and Installation
Clone the repository and install the required dependencies:

```bash
# Clone the repository
cd fake-account-detector

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Project (Recommended)
The project includes a unified script that handles data generation, model training, and server launch:

```bash
python setup_and_run.py
```
This will:
1. Generate `data/social_media_profiles.csv`.
2. Train the ML models and save the best one to `models/best_model.pkl`.
3. Launch the dashboard at **http://localhost:5000**.

---

## 🛠️ Manual Step-by-Step

If you prefer to run steps individually:

### Step 1: Data Generation
Generate the synthetic dataset of 4,000+ profiles:
```bash
python generate_dataset.py
```

### Step 2: Training & Evaluation
Train the ensemble of classifiers and generate performance plots:
```bash
python train_model.py
```
*Note: This generates confusion matrices, ROC curves, and feature importance plots in `static/plots/`.*

### Step 3: Launch Dashboard
Start the Flask web server:
```bash
python app.py
```

---

## 📊 How It Works

The system analyzes profiles across **6 feature groups**:
1. **Identity**: Profile picture, bio completeness, username characteristics.
2. **Activity**: Posting frequency, likes/comments per post, activity hours.
3. **Network**: Follower/following ratio, mutual friends, fake follower percentage.
4. **Behavioral**: Automation usage, hashtag density, mention frequency.
5. **Content**: Diversity score, original vs. repost ratio, spam keyword count.
6. **Engagement**: Average response time and engagement rates.

The training pipeline uses **SMOTE** (Synthetic Minority Over-sampling Technique) to handle class imbalances and selects the best performing model based on **ROC-AUC** scores.

---

## 📜 License
This project is open-source and available under the MIT License.
