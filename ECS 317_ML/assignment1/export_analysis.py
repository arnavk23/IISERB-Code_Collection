"""Export an analysis HTML report by running the analysis steps and saving figures.

This script reproduces the notebook EDA and model evaluation and writes
`analysis_and_report.html` along with image assets.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os

ROOT = Path.cwd()
DATA_DIR = ROOT / 'data'
OUT_HTML = ROOT / 'analysis_and_report.html'
IMG_DIR = ROOT / 'analysis_images'
IMG_DIR.mkdir(exist_ok=True)

np.random.seed(42)

def load_data():
    train = pd.read_csv(DATA_DIR / 'Training dataset.csv')
    test = pd.read_csv(DATA_DIR / 'Test data.csv')
    return train, test


def save_histograms(X: pd.DataFrame):
    imgs = []
    for col in X.columns:
        fig, ax = plt.subplots(figsize=(6,3))
        sns.histplot(X[col], ax=ax, kde=True)
        ax.set_title(col)
        p = IMG_DIR / f"hist_{col.replace(' ','_')}.png"
        fig.tight_layout()
        fig.savefig(p)
        plt.close(fig)
        imgs.append(p.name)
    return imgs


def save_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('pred')
    ax.set_ylabel('true')
    ax.set_title('Confusion Matrix (validation)')
    p = IMG_DIR / 'confusion_matrix.png'
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return p.name


def save_feature_importance(model, columns):
    try:
        importances = model.feature_importances_
    except Exception:
        return None
    fi = pd.Series(importances, index=columns).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6,3))
    sns.barplot(x=fi.values, y=fi.index, ax=ax)
    ax.set_title('Feature importances (RandomForest)')
    p = IMG_DIR / 'feature_importance.png'
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return p.name


def main():
    train, test = load_data()
    X = train.drop(columns=['Product Quality'])
    y = train['Product Quality']

    # histograms
    hist_imgs = save_histograms(X)

    # load model and scaler if available
    model_path = ROOT / 'best_model.joblib'
    scaler_path = ROOT / 'scaler.joblib'
    model = None
    scaler = None
    if model_path.exists() and scaler_path.exists():
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

    # create validation split and evaluate
    Xs = X.values
    if scaler is not None:
        Xs = scaler.transform(X)
    Xtr, Xval, ytr, yval = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)

    eval_report = ''
    cm_img = None
    fi_img = None
    if model is not None:
        preds = model.predict(Xval)
        eval_report = classification_report(yval, preds)
        cm_img = save_confusion_matrix(yval, preds)
        fi_img = save_feature_importance(model, X.columns)
    else:
        eval_report = 'No saved model found. Run train_and_tune.py to produce best_model.joblib.'

    # write HTML
    html = [
        '<html><head><meta charset="utf-8"><title>Analysis report</title></head><body>',
        '<h1>Assignment 1 — Analysis and Report</h1>',
        '<h2>Dataset overview</h2>',
        f'<p>Train shape: {train.shape}</p>',
        f'<p>Test shape: {test.shape}</p>',
        '<h2>Feature distributions</h2>'
    ]

    for img in hist_imgs:
        html.append(f'<img src="analysis_images/{img}" style="max-width:700px"><br>')

    html.append('<h2>Validation evaluation</h2>')
    html.append('<pre>')
    html.append(eval_report)
    html.append('</pre>')

    if cm_img:
        html.append('<h3>Confusion matrix</h3>')
        html.append(f'<img src="analysis_images/{cm_img}" style="max-width:400px"><br>')
    if fi_img:
        html.append('<h3>Feature importances</h3>')
        html.append(f'<img src="analysis_images/{fi_img}" style="max-width:700px"><br>')

    html.append('<h2>Reproducibility</h2>')
    html.append('<p>Run <code>python train_and_tune.py</code> to reproduce model selection and final_submission.csv.</p>')
    html.append('</body></html>')

    OUT_HTML.write_text('\n'.join(html), encoding='utf-8')
    print('Wrote', OUT_HTML)


if __name__ == '__main__':
    main()
