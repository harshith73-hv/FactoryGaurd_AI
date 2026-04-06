# FactoryGaurd_AI - Predictive Maintenance using Machine Learning

Production-grade Machine Learning system to predict industrial machine failures using IoT sensor data

---

## 🚀 Project Overview

FactoryGuard AI is a predictive maintenance system designed to detect machine failures before they occur, enabling proactive maintenance and reducing downtime.

This project simulates a real-world industrial environment using sensor data (temperature, torque, wear, etc.) and applies advanced ML techniques to predict failures with high precision.

Key challenges:
- Imbalanced dataset
- Time-dependent patterns
- Need for high precision and recall balance

---

## 📊 Dataset

- 📍 Source: UCI AI4I Predictive Maintenance Dataset
- 🔗 https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

📌 Key Features:
- Air Temperature [K]
- Process Temperature [K]
- Rotational Speed [rpm]
- Torque [Nm]
- Tool Wear [min]
- Failure Types (TWF, HDF, PWF, OSF, RNF)

⚠️ Challenge:
Highly imbalanced dataset (~3% failures)

---

## ⚙️ ML Pipeline

## 1. Feature Engineering

Custom feature engineering was applied to improve model performance:

-  Rolling Mean & Standard Deviation  
-  Exponential Moving Average (EMA)  
-  Lag Features(t-1, t-2)  
-  One-hot encoding for categorical features  
-  Dropping irrelevant columns  

---

## 2. ⚖️ Handling Imbalance

Used :
- Applied Synthetic Minority Oversampling Technique(SMOTE) only on training folds
- Avoided data leakage
- Used class weighting in models

---

## 3. 🤖 Models Used

### a. Logistic Regression (Baseline)
- Linear model
- Limited performance on complex data

---

### b. Random Forest
- Ensemble model
- Captures non-linear patterns
    
---

### 3. XGBoost 
- Gradient Boosting algorithm
- Best performance
- Handles complex feature interactions

---

### 4. LightGBM 
- Fast boosting model
- Slightly lower performance than XGBoost

---

## 🔍 Model Optimization

Used **Optuna** for hyperparameter tuning:

- Tuned parameters like:
  - `n_estimators`
  - `max_depth`
  - `learning_rate`
  - `subsample`
  - `colsample_bytree`
  - `scale_pos_weight`
  -  `random_state`
  - `n_jobs`
  - `deterministic`
  - `eval_metric`
  - `num_leaves`
  - `min_child_samples`

- Used **Stratified K-Fold Cross Validation**
- Applied **SMOTE inside each fold**

---

## 🎯 Threshold Optimization

Instead of using default threshold (0.5):

- Used **Precision-Recall Curve**
- Selected threshold based on **maximum F1-score**

---

## 📈 Evaluation Metrics

- Precision
- Recall
- F1-score
- Confusion Matrix
- Precision-Recall AUC (PR-AUC [Primary Metric])

---

## 🧠 Model Explainability

Used **SHAP (SHapley Additive Explanations)**:

- Understand feature importance
- Explain model predictions
- Identify key factors causing failures

---

## 🏆 Final Model Performance (XGBoost)

Applying SMOTE, Optuna tuning, and threshold optimization significantly improved model performance:

- Reduced class imbalance impact  
- Improved precision-recall balance  
- Increased PR-AUC score  
- Enhanced model interpretability using SHAP  


| Metric | Value |
|------|------|
| Precision (Failure) | 0.83 |
| Recall (Failure) | 0.55 |
| F1-score | 0.66 |
| PR-AUC | 0.75 |

Final model achieves high precision (0.83) while maintaining reasonable recall (0.55), making it suitable for real-world predictive maintenance where false alarms are costl

---

## 📊 Confusion Matrix

[[1930 7]
[ 28 34]]


---
## 📸 Visual Results
  ![Confusion Matri](assets\Confusion_Matrix.png)
  ![PR Curve](assets\Precision_Recall_Curve.png)

---

## 💡 Key Insights

- XGBoost outperformed all models
- High precision → fewer false alarms
- Moderate recall → scope for improvement
- Feature engineering significantly improved results

---

## ⚠️ Limitations

- Recall can be improved further
- Rare failures are still difficult to detect
- Real-time deployment not implemented

---

## 🚀 Future Improvements

- Deep Learning models (LSTM for time series)
- Deploy model using API (Flask/FastAPI)
- Real-time monitoring system

---

## 🛠️ Tech Stack

- Python
- Scikit-learn
- XGBoost
- LightGBM
- Optuna
- SHAP
- Pandas, NumPy
- Matplotlib, Seaborn

---

## 📁 Project Structure
├── src/
| ├── feature_engineering.py
| ├── lightgbm_model_train.py
| ├── lightgbm_optuna_model.py
| ├── Lr_model_train.py
| ├── Rf_model_train.py
| ├── Rf_optuna_model_train.py
| ├── xgb_model_train.py
| └── xgb_optuna_model_train.py
├── models/
| ├── lightgbm_model.pkl
│ ├── logistic_model.pkl
│ ├── logistic_model.pkl
│ ├── random_forest_optuna.pkl
│ ├── random_forest.pkl
│ ├── xgboost_model.pkl
│ └── xgb_optuna_model.pkl
└── data/
└── data.csv

Model folder is not included due to size. You can train using:
python src/xgb_optuna_model_train.py
---


▶️ How to Run
# Clone repo
git clone https://github.com/your-username/FactoryGuard_AI.git

# Install dependencies
pip install -r requirements.txt

# Run training
python src/xgb_optuna_model_train.py

---

# 👩‍💻 Author

Sarada
Machine Learning Engineer (Intern)

---

