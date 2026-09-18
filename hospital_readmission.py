# Hospital Readmission Prediction
# Logistic Regression with L2 Regularization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report, ConfusionMatrixDisplay


# 1. Load dataset
df = pd.read_csv("diabetic_data.csv")

print("Dataset shape:", df.shape)
print(df.head())


# 2. Create target
# <30 = readmitted within 30 days
# >30 and NO = not readmitted within 30 days

df["target"] = (df["readmitted"] == "<30").astype(int)


# 3. Select relevant features
features = [
    "age", "gender", "time_in_hospital",
    "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient",
    "number_emergency", "number_inpatient",
    "number_diagnoses", "diag_1", "diag_2", "diag_3"
]

X = df[features].replace("?", np.nan)
y = df["target"]


# 4. Numeric and categorical features
num_cols = [
    "time_in_hospital", "num_lab_procedures",
    "num_procedures", "num_medications",
    "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses"
]

cat_cols = ["age", "gender", "diag_1", "diag_2", "diag_3"]


# 5. Preprocessing
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_cols),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])


# 6. Logistic Regression + L2
model = Pipeline([
    ("preprocessor", preprocessor),
    ("logistic", LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        max_iter=1000
    ))
])


# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 8. Train model
model.fit(X_train, y_train)


# 9. Predict probability and class
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)


# 10. ROC-AUC
auc = roc_auc_score(y_test, y_prob)
print("\nROC-AUC:", round(auc, 3))


# 11. Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 12. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

ConfusionMatrixDisplay(
    cm,
    display_labels=["No Readmission", "30-Day Readmission"]
).plot()

plt.title("Confusion Matrix")
plt.show()


# 13. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], "--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Hospital Readmission")

plt.legend()
plt.grid()
plt.show()


# 14. Clinical Error Analysis
tn, fp, fn, tp = cm.ravel()

print("\nClinical Error Analysis")
print("False Positives:", fp)
print("False Negatives:", fn)
print("False Negative Rate:", round(fn / (fn + tp), 3))
print("False Positive Rate:", round(fp / (fp + tn), 3))
