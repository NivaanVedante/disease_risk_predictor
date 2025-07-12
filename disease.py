import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

st.set_page_config(page_title="Disease Prediction Model", layout="centered")

# -------------------------
# Load Dataset
# -------------------------
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

X = df.drop('target', axis=1)
y = df['target']

# -------------------------
# Preprocess
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# Sidebar Model Selector
# -------------------------
st.title("🩺 Disease Risk Predictor")
st.markdown("Predict disease (e.g., cancer) risk based on patient health records.")

model_choice = st.sidebar.selectbox("Select ML Model", ["Logistic Regression", "Random Forest", "SVM", "XGBoost"])

if model_choice == "Logistic Regression":
    model = LogisticRegression()
elif model_choice == "Random Forest":
    model = RandomForestClassifier()
elif model_choice == "SVM":
    model = SVC(probability=True)
else:
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
y_probs = model.predict_proba(X_test_scaled)[:, 1]

# -------------------------
# Evaluation Output
# -------------------------
st.subheader("Model Performance")
st.text(classification_report(y_test, y_pred))

auc = roc_auc_score(y_test, y_probs)
fpr, tpr, _ = roc_curve(y_test, y_probs)

fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
ax.set_title("ROC Curve")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()
st.pyplot(fig)

# -------------------------
# Live Prediction (Optional UI)
# -------------------------
st.subheader("🔍 Try a Live Prediction")
sample_idx = st.slider("Select Test Sample Index", 0, len(X_test)-1, 0)
sample = X_test.iloc[sample_idx:sample_idx+1]
sample_scaled = scaler.transform(sample)
pred = model.predict(sample_scaled)[0]
prob = model.predict_proba(sample_scaled)[0][1]

st.write(f"**Prediction:** {'Benign' if pred == 1 else 'Malignant'} (Confidence: {prob*100:.2f}%)")
