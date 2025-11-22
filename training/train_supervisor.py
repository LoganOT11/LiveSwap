import pandas as pd
import joblib
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --- PATH SETUP ---
# Get project root so we can find the dataset folder
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
csv_path = os.path.join(project_root, "dataset", "score_data.csv")
model_save_path = os.path.join(project_root, "models", "supervisor.pkl")

def train_supervisor():
    print(f"Loading data from: {csv_path}")
    
    if not os.path.exists(csv_path):
        print("Error: score_data.csv not found. Run 'data_tools/record_scores.py' first!")
        return

    # 1. Load Data
    df = pd.read_csv(csv_path)
    print(f"   Loaded {len(df)} rows.")

    # 2. Define Features (X) and Target (y)
    # We must use the exact same columns we recorded
    feature_cols = [
        "fast", "slow", "exp", "fast_avg",
        "fast_std", "fast_delta",  "slow_delta"
    ]
    
    # Check if columns exist (in case you have old data mixed with new)
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns in CSV: {missing_cols}")
        print("   Recommendation: Delete score_data.csv and record fresh data.")
        return

    X = df[feature_cols]
    y = df["label"]

    # 3. Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Train Random Forest
    # n_estimators=100: Use 100 decision trees
    # max_depth=10: Prevent overfitting (memorizing noise)
    print("Training Random Forest Supervisor...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)

    # 5. Evaluate
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    print("-" * 30)
    print(f"Model Accuracy: {acc:.2%}")
    print("-" * 30)
    print("Detailed Report:")
    print(classification_report(y_test, preds, target_names=["Content", "Ad"]))

    # 6. Save
    joblib.dump(clf, model_save_path)
    print(f"Supervisor saved to: {model_save_path}")

if __name__ == "__main__":
    train_supervisor()
