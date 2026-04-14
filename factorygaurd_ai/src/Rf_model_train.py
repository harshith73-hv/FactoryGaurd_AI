import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc


from factorygaurd_ai.src.feature_engineering import load_and_process_data

from imblearn.over_sampling import SMOTE


#----------------------LOAD AND PROCESS DATA----------------------
df = load_and_process_data('data/data.csv')

print("Data loaded and processed successfully.\n")

#------------------------SPLIT INPUT AND OUTPUT--------------------
x = df.drop('Machine_failure', axis=1)
y = df['Machine_failure']

print("Data split into x and y\n")

#-------------------------TRAIN TEST SPLIT------------------------- 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Step 2: apply SMOTE ONLY on training data

smote = SMOTE(random_state=42)
x_train, y_train = smote.fit_resample(x_train, y_train)
print("Train Test split is done")

#-----------------------MODEL CREATION---------------------


rf_model = RandomForestClassifier(
    n_estimators=100, #number of trees

    random_state=42, #for reproducibility
    class_weight='balanced'
)


print("Random Forest Model Created\n")

#-----------------------MODEL TRAINING---------------------

rf_model.fit(x_train, y_train)

print("Model training is done\n")



#-----------------------MODEL PREDICTION (WITH THRESHOLD TUNING)---------------------
y_prob = rf_model.predict_proba(x_test)[:, 1]


# threshold = 0.6
# y_pred = (y_prob > threshold).astype(int)

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


#-----------------------MODEL EVALUATION---------------------
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

#-----------------------MODEL SAVING---------------------
joblib.dump(rf_model, "models/random_forest.pkl")   
print("Model saved successfully at 'models/random_forest.pkl'")