# Assignment 1 - Pipeline

This repository contains a pipeline for the Machine Learning assignment. It loads the provided training and test datasets, runs basic EDA, trains and tunes models, evaluates them, and produces a final submission CSV and a saved model artifact.

Files:
- `assignment1_pipeline.py` - quick pipeline: EDA, baseline training (LogisticRegression) and RandomForest, produces `test_predictions.csv`.
- `train_and_tune.py` - full training and hyperparameter tuning (GridSearchCV), saves `best_model.joblib`, `scaler.joblib`, `final_submission.csv`, and `report.txt`.
- `data/Training dataset.csv` - training data (provided).
- `data/Test data.csv` - test data (provided).
- `requirements.txt` - Python dependencies.

Quick reproduction
1. Create and activate a Python 3.8+ virtualenv:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the quick pipeline (fast):

```bash
python assignment1_pipeline.py
```

4. (Recommended) Run full train + tuning (this will take longer):

```bash
python train_and_tune.py
```

Outputs (for submission)
- `final_submission.csv` — predicted `Product Quality` for the test set (CSV, no index). Use this for uploading to the assignment portal.
- `best_model.joblib`, `scaler.joblib` — saved model and scaler used to make `final_submission.csv`.
- `report.txt` — short model selection report with validation metrics.

Notes & grading tips
- The Random Forest model is hyperparameter tuned with GridSearchCV (f1_weighted). The script saves the best estimator and scaler so graders can reproduce predictions.
- If you need a notebook-style report with plots, I can add a Jupyter notebook with EDA plots, confusion matrices, and model interpretation.

Contact
If you want me to further tune, add plots, or prepare a short PDF report (1-2 pages) summarizing methods and results for submission, tell me which you'd prefer and I'll add it.
