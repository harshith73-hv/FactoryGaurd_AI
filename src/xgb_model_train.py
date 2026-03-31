import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

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


# ---------------------- HANDLE IMBALANCE ----------------------
smote = SMOTE(random_state=42)
x_train, y_train = smote.fit_resample(x_train, y_train)

print("SMOTE applied\n")


# ---------------------- MODEL CREATION ----------------------
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

print("XGBoost model created\n")


# ---------------------- TRAIN MODEL ----------------------
xgb_model.fit(x_train, y_train)

print("Model training completed\n")


# ---------------------- PREDICTION ----------------------
y_prob = xgb_model.predict_proba(x_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

# threshold tuning
f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)

best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]

print(f"Best F1-score: {f1_scores[best_index]:.4f}")

print(f"Best Threshold based on F1-score: {best_threshold:.4f}")
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
joblib.dump(xgb_model, "models/xgboost_model.pkl")

print("Model saved successfully at 'models/xgboost_model.pkl'")