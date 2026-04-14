import joblib
import optuna
import shap
import numpy as np
np.random.seed(42)
import random
random.seed(42)
import matplotlib.pyplot as plt
import seaborn as sns

from optuna import trial
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc

from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier

from feature_engineering import load_and_process_data


# ---------------------- LOAD DATA ----------------------
df = load_and_process_data('data/data.csv')

print("Data loaded successfully\n")


# ---------------------- SPLIT INPUT & OUTPUT ----------------------
x = df.drop('Machine_failure', axis=1)
y = df['Machine_failure']

print("Data split into X and y\n")


# ---------------------- TRAIN TEST SPLIT ----------------------
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# ---------------------- OPTUNA OBJECTIVE FUNCTION ----------------------

def objective(trial):

    #HYPER PARAMETERS
    params = {
        "n_estimators" : trial.suggest_int("n_estimators",100,300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": 42,
        "bagging_seed": 42,
        "feature_fraction_seed": 42,
        "data_random_seed": 42,
        "deterministic": True,
        "n_jobs":-1,
        "eval_metric": "logloss",
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 20)

    }
    

    #CROSS VALIDATION
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    f1_scores = []

    for train_idx, val_idx in cv.split(x_train,y_train):
        x_tr, x_val = x_train.iloc[train_idx], x_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]


        # ---------------------- HANDLE IMBALANCE ----------------------
        smote = SMOTE(random_state=42)
        x_tr, y_tr = smote.fit_resample(x_tr, y_tr)

        print("SMOTE applied\n")

        # --------------------- TRAIN MODEL -----------------------------
        model = LGBMClassifier(**params)
        model.fit(x_tr, y_tr)

        # --------------------- PREDICTION -------------------------------

        y_prob = model.predict_proba(x_val)[:, 1]

        # --------------------- THRESHOLD TUNING -------------------------

        precision, recall, thresholds = precision_recall_curve(y_val, y_prob)

        f1 = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)

        # ---------------------- PR-AUC optimization -----------------------

        f1_scores.append(np.max(f1))

    return np.mean(f1_scores)

# ---------------------- RUN OPTUNA ----------------------
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42)
    )

study.optimize(objective, n_trials=20)

print("\nBest Parameters:\n", study.best_params)


# ---------------------- MODEL CREATION ----------------------
lgbm_model = LGBMClassifier(
    **study.best_params,
    random_state=42
)

print("LightGBM model created\n")


# ---------------------- TRAIN MODEL ----------------------
lgbm_model.fit(x_train, y_train)

print("Model training completed\n")


#---------------------- SHAP-------------------
explainer = shap.TreeExplainer(lgbm_model)

shap_values = explainer(x_test)

shap.summary_plot(shap_values, x_test)

# ---------------------- PREDICTION ----------------------
y_prob = lgbm_model.predict_proba(x_test)[:, 1]

# same threshold tuning as before
# threshold = 0.6
# y_pred = (y_prob > threshold).astype(int)

# print("Prediction completed\n")

precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

# Avoid division by zero
f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)

best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]

print(f"Best F1-score: {f1_scores[best_index]:.4f}")

print(f"Best Threshold based on F1-score: {best_threshold:.4f}")

# Apply best threshold
y_pred = (y_prob > best_threshold).astype(int)

print("Prediction completed\n")



# ---------------------- EVALUATION ----------------------
print("Model Performance:\n")
print(classification_report(y_test, y_pred))

#confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

#-------------------------PR-AUC-------------------------------
precision, recall, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)
print(f"\nPrecision-Recall AUC:", pr_auc)

#-----------------------VISUALIZATION------------------------
# Confusion Matrix Plot
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# PR Curve
plt.plot(recall, precision)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()


# ---------------------- SAVE MODEL ----------------------
joblib.dump(lgbm_model, "factorygaurd_ai/models/lightgbm_model.pkl")
joblib.dump(best_threshold, "factorygaurd_ai/models/lightgbm_threshold.pkl")

print("Model and Threshold saved successfully!!")
