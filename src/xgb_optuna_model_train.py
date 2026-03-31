import joblib
import optuna
import shap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, f1_score, confusion_matrix, precision_recall_curve, auc, average_precision_score

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from feature_engineering import load_and_process_data


# ---------------------- LOAD DATA ----------------------
df = load_and_process_data('data/data.csv')

x = df.drop('Machine_failure', axis=1)
y = df['Machine_failure']

# ---------------------- TRAIN TEST SPLIT ----------------------
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# ---------------------- OPTUNA OBJECTIVE ----------------------
def objective(trial):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": 42,
        "eval_metric": "logloss",
        "gamma": trial.suggest_float("gamma", 0, 5),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 20)
    }

    model = XGBClassifier(**params)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    f1_scores = []

    for train_idx, val_idx in cv.split(x_train, y_train):
        x_tr, x_val = x_train.iloc[train_idx], x_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # ---------------------- SMOTE ----------------------
        smote = SMOTE(random_state=42, sampling_strategy='auto')
        x_tr, y_tr = smote.fit_resample(x_tr, y_tr)

        #----------------------- TRAIN MODEL ----------------------

        model.fit(x_tr, y_tr)

        #---------------------- PREDICTION ----------------------

        y_prob = model.predict_proba(x_val)[:, 1]


        #-------------------- Threshold tuning inside CV ---------------
        precision, recall, thresholds = precision_recall_curve(y_val, y_prob)

        f1 = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)

        #-------------------- PR-AUC optimization inside CV ---------------
        # score = average_precision_score(y_val, y_prob)

        f1_scores.append(np.max(f1))

    return np.mean(f1_scores)

    #model.fit(x_train, y_train)

    #y_pred = model.predict(x_test)

    #return f1_score(y_test, y_pred)


# ---------------------- RUN OPTUNA ----------------------
study = optuna.create_study(direction="maximize")

study.optimize(objective, n_trials=20)

print("\nBest Parameters:\n", study.best_params)


# ---------------------- TRAIN FINAL MODEL ----------------------
best_model = XGBClassifier(
    **study.best_params,
    eval_metric='logloss'
    )

best_model.fit(x_train, y_train)

#---------------------- SHAP-------------------
explainer = shap.TreeExplainer(best_model)

shap_values = explainer(x_test)

shap.summary_plot(shap_values, x_test)

# ---------------------- PREDICTION ----------------------
y_prob = best_model.predict_proba(x_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"Best F1-score: {f1_scores[best_idx]:.4f}")

print(f"Best Threshold: {best_threshold:.4f}")

y_pred = (y_prob > best_threshold).astype(int)

# ---------------------- EVALUATION ----------------------
print("\nFinal Model Performance:\n")
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
joblib.dump(best_model, "models/xgb_optuna_model.pkl")

print("\nModel saved successfully!")