"""
TradeGuard AI — Trade Risk Classifier
----------------------------------------
Trains a RandomForest classifier to predict whether a proposed trade will
be 'approved' or 'rejected', using features derivable at proposal time
(no leakage from the rules engine's own output).

Honest framing: trained primarily on rule-grounded synthetic data (see
generate_training_data.py) with a handful of real completed trades mixed
in as anchors. This is NOT a model of real-world negotiation outcomes —
it's a learned approximation of CBA salary-matching logic from features,
evaluated the same way any classifier should be: train/test split,
accuracy, precision, recall, F1, ROC-AUC.

Run:  python3 ml/train_model.py
"""

import csv
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.preprocessing import LabelEncoder

BASE = Path(__file__).resolve().parent.parent
DATA_PATH = BASE / "data" / "raw" / "trade_history_seed.csv"
MODEL_PATH = Path(__file__).resolve().parent / "risk_model.pkl"

APRON_ORDER = ["under_cap", "over_cap", "over_tax", "first_apron", "second_apron"]


def load_data():
    with open(DATA_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def build_features(rows):
    apron_encoder = {tier: i for i, tier in enumerate(APRON_ORDER)}

    X, y, sources = [], [], []
    for r in rows:
        apron_rank = apron_encoder.get(r["team_from_apron_status"], 0)
        X.append([
            apron_rank,
            int(r["n_outgoing_contracts"]),
            float(r["outgoing_total"]),
            float(r["incoming_total"]),
            float(r["salary_ratio"]),
        ])
        y.append(1 if r["outcome"] == "approved" else 0)  # 1 = approved, 0 = rejected/flagged
        sources.append(r["source"])
    return np.array(X), np.array(y), sources


def main():
    rows = load_data()
    X, y, sources = build_features(rows)
    feature_names = ["apron_tier_rank", "n_outgoing_contracts", "outgoing_total",
                      "incoming_total", "salary_ratio"]

    X_train, X_test, y_train, y_test, src_train, src_test = train_test_split(
        X, y, sources, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print("=== Evaluation on held-out test set ===")
    print(f"Test set size: {len(y_test)}  (includes {sum(1 for s in src_test if s=='real')} real trades)")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
    print(f"F1:        {f1_score(y_test, y_pred):.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.3f}")
    print()
    print("Confusion matrix (rows=actual, cols=predicted) [rejected, approved]:")
    print(confusion_matrix(y_test, y_pred))
    print()
    print(classification_report(y_test, y_pred, target_names=["rejected", "approved"]))

    print("=== Feature importances (RandomForest built-in — SHAP unavailable offline) ===")
    importances = sorted(zip(feature_names, clf.feature_importances_), key=lambda x: -x[1])
    for name, imp in importances:
        print(f"  {name:20s} {imp:.3f}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": clf, "feature_names": feature_names, "apron_order": APRON_ORDER}, f)
    print(f"\nModel saved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
