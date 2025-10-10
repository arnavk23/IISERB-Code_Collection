"""Train and tune models for Assignment 1

Performs cross-validated hyperparameter search for RandomForest and LogisticRegression,
saves the best model, writes final predictions (named final_submission.csv), and produces a short text report.
"""
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = ROOT


def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, "Training dataset.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "Test data.csv"))
    return train, test


def main():
    train, test = load_data()
    X = train.drop(columns=["Product Quality"])
    y = train["Product Quality"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(test)

    # Grid search for RandomForest
    rf = RandomForestClassifier(random_state=42)
    rf_params = {
        "n_estimators": [100, 200, 400],
        "max_depth": [None, 6, 10],
        "min_samples_split": [2, 5],
    }
    print("Running GridSearchCV for RandomForest (this may take a minute)...")
    gs_rf = GridSearchCV(rf, rf_params, cv=5, scoring="f1_weighted", n_jobs=-1)
    gs_rf.fit(X_train_s, y_train)
    print("Best RF params:", gs_rf.best_params_)

    # Grid search for Logistic Regression
    lr = LogisticRegression(max_iter=2000, random_state=42)
    lr_params = {"C": [0.01, 0.1, 1, 10]}
    gs_lr = GridSearchCV(lr, lr_params, cv=5, scoring="f1_weighted", n_jobs=-1)
    gs_lr.fit(X_train_s, y_train)
    print("Best LR params:", gs_lr.best_params_)

    # Evaluate both on validation
    models = {
        "rf": gs_rf.best_estimator_,
        "lr": gs_lr.best_estimator_,
    }
    results = {}
    for name, model in models.items():
        preds = model.predict(X_val_s)
        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, average="weighted")
        print(f"{name} val acc: {acc:.4f}, f1: {f1:.4f}")
        results[name] = (model, acc, f1)

    best_name = max(results.items(), key=lambda x: x[1][2])[0]
    best_model = results[best_name][0]
    print(f"Selected best model: {best_name}")

    # Save model and scaler
    model_path = os.path.join(OUT_DIR, "best_model.joblib")
    scaler_path = os.path.join(OUT_DIR, "scaler.joblib")
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Saved model to {model_path} and scaler to {scaler_path}")

    # Create final predictions and save as final_submission.csv (grading-ready)
    final_preds = best_model.predict(X_test_s)
    out = pd.DataFrame({"Product Quality": final_preds})
    out_path = os.path.join(OUT_DIR, "final_submission.csv")
    out.to_csv(out_path, index=False)
    print(f"Wrote final submission to {out_path}")

    # Short report
    report_path = os.path.join(OUT_DIR, "report.txt")
    with open(report_path, "w") as f:
        f.write("Model selection report\n")
        f.write(f"Best model: {best_name}\n")
        f.write(str(results[best_name][0]) + "\n")
        f.write("Validation accuracy: %.4f\n" % results[best_name][1])
        f.write("Validation f1 (weighted): %.4f\n" % results[best_name][2])
    print(f"Wrote short report to {report_path}")


if __name__ == "__main__":
    main()
