# Heart Disease XGBoost Classifier

- Model ID: adarshsalukhe/heart-disease-xgboost-classifier
- License: Apache 2.0
- Tasks: healthcare, beginner, advanced, classification, tabular
- Frameworks: Other
- Shared by: Adarsh Salukhe
- Source: https://www.kaggle.com/models/adarshsalukhe/heart-disease-xgboost-classifier

## Description

Overview
This is an XGBoost binary classifier trained to predict presence of heart disease from clinical and demographic features.

Training Data
Trained on the Heart Failure Prediction Dataset (918 rows, 11 features), which combines five heart disease datasets (Cleveland, Hungary, Switzerland, VA, Stalog). Features include age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, max heart rate, exercise-induced angina, oldpeak, and ST slope. Target is binary: heart disease present (1) or absent (0).
Model Details

Algorithm: XGBoost (XGBClassifier)
Hyperparameters: n_estimators=100, max_depth=3, learning_rate=0.1
No hyperparameter tuning applied this is a baseline model
Data was already numeric/pre-encoded, so no additional preprocessing pipeline was required

Performance

Single train/test split (80/20, stratified): 97.6% accuracy, 97.6% balanced accuracy
5-fold stratified cross-validation (more reliable estimate): 95.8% mean balanced accuracy (std dev 0.022)
Recall on positive class (disease present): 0.97 model misses roughly 3% of true disease cases

Limitations

Small dataset (918 rows) means performance estimates have some variance across folds (std ~0.022)
No hyperparameter tuning performed this is an untuned baseline
Not validated on external/out-of-distribution clinical data
Not intended for real clinical or diagnostic use this is an educational/portfolio project

Intended Use
Educational demonstration of XGBoost for tabular binary classification. Not for medical decision-making.

## References

This model was created from scratch by training an XGBoost classifier on the publicly available Heart Failure Prediction Dataset by fedesoriano on Kaggle, which itself combines five original heart disease datasets (Cleveland, Hungary, Switzerland, VA Long Beach, and Stalog) collected via clinical cardiology studies.
Training was performed independently: data was split 80/20 (stratified) for evaluation, with 5-fold stratified cross-validation used to confirm performance stability. No pretrained weights, existing model checkpoints, or third-party model artifacts were used — the model was trained end-to-end from the raw tabular data using xgboost.XGBClassifier with baseline (untuned) hyperparameters.
No external code, notebooks, or model files were copied; the preprocessing, training, and evaluation pipeline was written and run independently as part of a personal learning project.
