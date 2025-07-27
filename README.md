# 💼 Salary Prediction Using Machine Learning

This project predicts a person's salary category (`<=50K` or `>50K`) using various machine learning algorithms based on demographic and professional attributes such as age, gender, education, job title, experience, and work hours.

An interactive **Streamlit web application** is included for real-time prediction and model exploration.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Model Training](#model-training)
- [Download Model Files](#download-model-files)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Results](#results)
- [Future Scope](#future-scope)
- [References](#references)
- [Contact](#contact)

---

## 📝 Overview

This project builds a classification system to determine whether a person earns more than $50,000 per year. It uses multiple ensemble models (Random Forest, XGBoost, LightGBM, CatBoost) and a stacking classifier to boost accuracy. 

It also offers a clean Streamlit interface for prediction and model switching.

---

## ✨ Features

- 📊 Predicts salary category (`<=50K` or `>50K`)
- 🧠 Supports 10+ ML models including CatBoost, LightGBM, XGBoost
- 🖥️ Interactive Streamlit UI
- 📁 Model persistence using `.pkl` files
- 🔍 Feature importance & accuracy visualization
- 📦 Clean project structure, easy to deploy

---

## 🧠 Model Training

Trained using the UCI Adult Dataset and the following models:
- Logistic Regression
- Decision Tree
- Random Forest
- Naive Bayes
- K-Nearest Neighbors
- MLPClassifier
- XGBoost
- LightGBM
- CatBoost
- HistGradientBoostingClassifier
- Stacking Classifier (final estimator: Logistic Regression)

Each model was trained after:
- Data preprocessing & encoding
- Handling null values
- Feature scaling & selection

---

## 🔗 Download Model Files

- 📦 **All `.pkl` model files**:  
  [📂 Google Drive – All Trained Models](https://drive.google.com/drive/folders/1wsKGvO10JQuf6rrq8pMtm-uHsJvVlY69?usp=sharing)

- 🐱 **CatBoost-related files only**:  
  [📂 Google Drive – CatBoost Files](https://drive.google.com/drive/folders/1Yqvbt4QhAGvdiAyJGMDGj8DAGBo1oPdw?usp=drive_link)

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/salary-prediction-ml.git
cd salary-prediction-ml
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
🚀 Usage
To run the Streamlit app:

bash
Copy
Edit
streamlit run app.py
You’ll be prompted to enter:

Age

Gender

Marital Status

Job Title

Years of Experience

Education Level

Education (as numeric level)

Hours Per Week

City and Location Type

Nationality

➡️ The app will predict your salary class and show the selected model's confidence.

🛠️ Technologies Used
Python 3.10+

Pandas, NumPy

Scikit-learn

XGBoost

LightGBM

CatBoost

Streamlit

Matplotlib, Seaborn

📊 Results
Model	Accuracy
Logistic Regression	78%
Random Forest	84%
LightGBM	85%
XGBoost	86%
CatBoost	88%
Stacking Classifier	89%
