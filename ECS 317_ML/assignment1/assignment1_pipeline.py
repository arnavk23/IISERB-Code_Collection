"""
Assignment 1 pipeline

Loads training and test CSVs, does basic EDA, trains two classifiers (LogisticRegression and
RandomForest), evaluates on a validation split, and writes test predictions to CSV.

Usage: run in the repository root. Requires packages in requirements.txt
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report


ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT, "data")


def load_data():
    train_path = os.path.join(DATA_DIR, "Training dataset.csv")
    test_path = os.path.join(DATA_DIR, "Test data.csv")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def basic_eda(df, name="data"):
    print(f"\n=== EDA: {name} ===")
    print(df.head())
    print("shape:", df.shape)
    print(df.describe())
    print("missing:")
    print(df.isnull().sum())


def prepare(train):
    X = train.drop(columns=["Product Quality"]) if "Product Quality" in train.columns else train
    y = train["Product Quality"] if "Product Quality" in train.columns else None
    return X, y


def run():
    train, test = load_data()
    basic_eda(train, "train")
    basic_eda(test, "test")

    X, y = prepare(train)

    # split for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(test)

    models = {
        "logreg": LogisticRegression(max_iter=1000, random_state=42),
        "rf": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_s, y_train)
        preds = model.predict(X_val_s)
        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, average="weighted")
        print(f"{name} val accuracy: {acc:.4f}, f1: {f1:.4f}")
        print(classification_report(y_val, preds))
        results[name] = (model, acc, f1)

    # choose best by f1
    best_name = max(results.items(), key=lambda x: x[1][2])[0]
    best_model = results[best_name][0]
    print(f"\nBest model: {best_name}")

    # predict on test set
    test_preds = best_model.predict(X_test_s)
    out = pd.DataFrame({"Product Quality": test_preds})
    out_path = os.path.join(ROOT, "test_predictions.csv")
    out.to_csv(out_path, index=False)
    print(f"Wrote test predictions to {out_path}")


if __name__ == "__main__":
    run()
