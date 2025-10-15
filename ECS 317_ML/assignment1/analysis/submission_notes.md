Submission notes
================

Files included in the submission bundle `assignment1_submission.zip`:

- `final_submission.csv`: Predicted `Product Quality` for the test dataset (CSV, no index). Use this for portal upload.
- `best_model.joblib`: Saved scikit-learn estimator (RandomForest) selected by validation F1.
- `scaler.joblib`: Saved StandardScaler used to transform features before prediction.
- `report.txt`: Short model selection report with validation metrics and selected hyperparameters.
- `README.md`: Reproduction instructions and notes.
- `train_and_tune.py`: The script used to tune models and generate final predictions.

How to reproduce locally
------------------------
1. Create & activate a virtualenv, install `requirements.txt`.
2. Run `python train_and_tune.py` to reproduce the final model and `final_submission.csv`.

Notes for grader
---------------
- The submitted `final_submission.csv` was produced by the hyperparameter-tuned Random Forest (GridSearchCV, scoring=f1_weighted).
- If you want to re-evaluate or reproduce using a different random seed or CV scheme, run `train_and_tune.py` and consult `report.txt` for the selected parameters.
