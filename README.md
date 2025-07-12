# 🩺 Disease Risk Predictor — CodeAlpha Internship Task 4

This project was developed as part of the **CodeAlpha Machine Learning Internship**, specifically for **Task 4: Disease Prediction from Medical Data**. It uses multiple machine learning models to predict the likelihood of a disease based on structured health data such as age, cell characteristics, and test results.

---

## 🎯 Objective

To build a predictive system that can classify whether a patient is at risk of a disease (e.g., breast cancer) using diagnostic data. The model is deployed via an interactive **Streamlit** web application.

---

## 📊 Dataset

We use the **Breast Cancer Wisconsin Diagnostic Dataset** available through `sklearn.datasets`. This dataset contains 30 real-valued features derived from digitized images of fine needle aspirate (FNA) of breast masses.

- **Samples:** 569
- **Features:** 30 (e.g., mean radius, mean texture, mean area, etc.)
- **Target:**  
  `0 = Malignant`  
  `1 = Benign`

---

## 🧠 Machine Learning Models

The app supports multiple classifiers that can be selected via the sidebar:
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- XGBoost (optional)

---

## 📈 Features

- 🔎 Live prediction on test samples
- 📊 ROC Curve & AUC score
- 🧾 Classification report for precision, recall, F1-score
- 🖥️ Web UI built with Streamlit
- 💬 Model selection and test sample interaction
