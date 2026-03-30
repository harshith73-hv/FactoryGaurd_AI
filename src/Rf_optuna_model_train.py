import joblib
import optuna
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from optuna import trial
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
from feature_engineering import load_and_process_data
from imblearn.over_sampling import SMOTE


# ---------------------- LOAD DATA ----------------------
df = load_and_process_data('data/data.csv')
print("Data loaded and processed successfully.\n")


# ---------------------- SPLIT INPUT & OUTPUT ----------------------
x = df.drop('Machine_failure', axis=1)
y = df['Machine_failure']
print("Data split into x and y\n")


# ---------------------- TRAIN TEST SPLIT ----------------------
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)


# ---------------------- OPTUNA OBJECTIVE FUNCTION ----------------------
def objective(trial):

    # hyper parameters
    n_estimators = trial.suggest_int('n_estimators', 100, 320)
    max_depth = trial.suggest_int('max_depth', 5, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 30)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

    # Create model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        class_weight='balanced',
        random_state=42
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    f1_scores = []
    
    for train_idx, val_idx in skf.split(x_train,y_train):

        x_tr, x_val = x_train.iloc[train_idx], x_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # ---------------------- APPLY SMOTE ----------------------
        smote = SMOTE(random_state=42)
        x_tr, y_tr = smote.fit_resample(x_tr, y_tr)


        # Train
        model.fit(x_tr, y_tr)

        # Predict probabilities
        y_prob = model.predict_proba(x_val)[:, 1]

    # # Apply threshold
    # threshold = 0.2
    # y_pred = (y_prob > threshold).astype(int)

    # # Evaluate using F1 score
    # return f1_score(y_test, y_pred)

  

        precision, recall, thresholds = precision_recall_curve(y_val, y_prob)

        f1_score = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)

        f1_scores.append(np.max(f1_score))

    return np.mean(f1_scores)


# ---------------------- RUN OPTUNA ----------------------
print("Running Optuna optimization...\n")

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=25)

print("Best Parameters:", study.best_params)


# ---------------------- TRAIN FINAL MODEL ----------------------
best_params = study.best_params

best_model = RandomForestClassifier(
    **best_params,
    class_weight='balanced',
    random_state=42
)

best_model.fit(x_train, y_train)

print("\nFinal model trained with best parameters\n")


# ---------------------- PREDICTION ----------------------
y_prob = best_model.predict_proba(x_test)[:, 1]

# threshold = 0.7
# y_pred = (y_prob > threshold).astype(int)

# print("Prediction completed\n")

precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"Best Threshold: {best_threshold:.4f}")

y_pred = (y_prob > best_threshold).astype(int)


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
joblib.dump(best_model, "models/random_forest_optuna.pkl")
print("Model saved successfully at 'models/random_forest_optuna.pkl'")                                             