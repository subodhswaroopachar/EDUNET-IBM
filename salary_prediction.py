
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import json
import os

# Import set_output for scikit-learn 1.2+ to make transformers output pandas DataFrames
from sklearn import set_config
set_config(transform_output="pandas")

# --- PAGE CONFIG ---
st.set_page_config(page_title="Salary & Income Prediction", layout="wide")

# --- TITLE ---
st.title("💼 Salary & Income Prediction App")
st.write("Predict whether an individual's income is **<=50K** or **>50K**, or estimate their numerical salary based on demographic and employment data.")
st.markdown("---")

# --- CustomPreprocessor ---
class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoders = {}
        self.fitted_categories = {}

    def fit(self, X, y=None):
        categorical_columns = ['workclass', 'marital-status', 'occupation', 'relationship',
                              'race', 'gender', 'native-country']
        X_copy = X.copy()
        for col in categorical_columns:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].replace('?', 'Others')
                le = LabelEncoder()
                le.fit(X_copy[col].astype(str))
                self.label_encoders[col] = le
                self.fitted_categories[col] = list(le.classes_)
        return self

    def transform(self, X):
        X_transformed = X.copy()
        if 'workclass' in X_transformed.columns:
            X_transformed['workclass'] = X_transformed['workclass'].replace('?', 'Others')
        if 'occupation' in X_transformed.columns:
            X_transformed['occupation'] = X_transformed['occupation'].replace('?', 'Others')
        if 'native-country' in X_transformed.columns:
            X_transformed['native-country'] = X_transformed['native-country'].replace('?', 'Others')
        if 'education' in X_transformed.columns:
            X_transformed = X_transformed.drop(columns=['education'], errors='ignore')
        for col, le in self.label_encoders.items():
            if col in X_transformed.columns:
                unseen_values_mask = ~X_transformed[col].isin(le.classes_)
                if unseen_values_mask.any():
                    X_transformed.loc[unseen_values_mask, col] = 'Others'
                X_transformed[col] = le.transform(X_transformed[col].astype(str))
            else:
                st.warning(f"Column '{col}' not found in input data during preprocessing. Skipping encoding.")
        return pd.DataFrame(X_transformed, columns=X_transformed.columns)

# --- Model and Metadata Loading for Income Prediction ---
INCOME_MODEL_PATH = r"C:\Users\subod\Downloads\salary_prediction\trained_models\best_income_prediction_pipeline.pkl"
INCOME_NUM_COLS_PATH = r"C:\Users\subod\Downloads\salary_prediction\trained_models\numerical_columns.json"
INCOME_CAT_COLS_PATH = r"C:\Users\subod\Downloads\salary_prediction\trained_models\categorical_columns.json"

@st.cache_resource
def load_income_assets():
    errors = []
    if not os.path.exists(INCOME_MODEL_PATH):
        errors.append(f"Model file not found: {INCOME_MODEL_PATH}")
    if not os.path.exists(INCOME_NUM_COLS_PATH):
        errors.append(f"Numerical columns file not found: {INCOME_NUM_COLS_PATH}")
    if not os.path.exists(INCOME_CAT_COLS_PATH):
        errors.append(f"Categorical columns file not found: {INCOME_CAT_COLS_PATH}")
    
    if errors:
        for error in errors:
            st.error(error)
        st.error("Please run 'train_model.py' to generate missing income prediction files.")
        st.stop()
    
    try:
        pipeline = joblib.load(INCOME_MODEL_PATH)
        with open(INCOME_NUM_COLS_PATH, 'r') as f:
            numerical_columns = json.load(f)
        with open(INCOME_CAT_COLS_PATH, 'r') as f:
            categorical_columns = json.load(f)
        st.success("Income prediction model and metadata loaded successfully!")
        return pipeline, numerical_columns, categorical_columns
    except Exception as e:
        st.error(f"Error loading income prediction assets: {e}")
        st.stop()

# Load income assets
income_assets = load_income_assets()
best_pipeline, numerical_columns, categorical_columns = income_assets

# --- Model Options for Salary Prediction ---
SALARY_MODEL_DIR = r"C:\Users\subod\Downloads\salary_prediction\trained_models"
MODEL_FILES = {
    "Random Forest": "random_forest_pipeline.pkl",
    "K-Nearest Neighbors": "knn_pipeline.pkl",
    "Decision Tree": "decision_tree_pipeline.pkl",
    "Extra Trees": "extra_trees_pipeline.pkl",
    "Logistic Regression": "logistic_regression_pipeline.pkl",
    "MLP Regressor": "mlp_classifier_pipeline.pkl",
    "XGBoost": "xgboost_pipeline.pkl",
    "LightGBM": "lightgbm_pipeline.pkl",
    "CatBoost": "catboost_pipeline.pkl",
    "HistGradientBoosting": "histgradientboosting_pipeline.pkl",
    "Stacking": "stacking_pipeline.pkl",
    "Naive Bayes": "naive_bayes_pipeline.pkl"
}
# Dynamically filter available models
model_options = {}
for name, file_name in MODEL_FILES.items():
    model_path = os.path.join(SALARY_MODEL_DIR, file_name)
    if os.path.exists(model_path):
        # No accuracy for now (placeholder removed)
        model_options[name] = (model_path, None)
    else:
        st.warning(f"Model file {file_name} not found at {model_path}. Skipping {name} model.")

if not model_options:
    st.error("No salary prediction models found in the directory. Please generate the model files and try again.")
    st.stop()

# --- Load Salary Model ---
@st.cache_resource
def load_salary_model(model_path):
    try:
        st.write(f"Attempting to load model from: {model_path}")
        if not os.path.exists(model_path):
            st.error(f"Salary model file not found: {model_path}")
            st.stop()
        model = joblib.load(model_path)
        st.success(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        st.error(f"Failed to load salary model from {model_path}: {e}")
        st.stop()

# --- Load Original Data for Income Prediction Dropdowns ---
@st.cache_data
def load_original_data_for_options():
    csv_file_path = r"C:\Users\subod\Downloads\salary_prediction\adult 3.csv"
    if not os.path.exists(csv_file_path):
        st.warning(f"Original dataset for dropdown options not found at '{csv_file_path}'. Using limited default values.")
        return pd.DataFrame(columns=['age', 'workclass', 'fnlwgt', 'education', 'educational-num', 'marital-status',
                                    'occupation', 'relationship', 'race', 'gender', 'capital-gain', 'capital-loss',
                                    'hours-per-week', 'native-country', 'income'])
    try:
        data = pd.read_csv(csv_file_path)
        data.columns = data.columns.str.strip()
        for col in data.select_dtypes(include=['object']).columns:
            data[col] = data[col].str.strip()
        data['workclass'] = data['workclass'].replace('?', 'Others')
        data = data[~data['workclass'].isin(['Without-pay', 'Never-worked'])]
        data['education'] = data['education'].replace('?', 'Others')
        data = data[~data['education'].isin(['1st-4th', '5th-6th', 'Preschool'])]
        data = data[(data['age'] <= 75) & (data['age'] >= 17)]
        for col in ['marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']:
            if col in data.columns and '?' in data[col].unique():
                data[col] = data[col].replace('?', 'Others')
        if 'income' in data.columns:
            data['income'] = data['income'].replace({'<=50K.': '<=50K', '>50K.': '>50K'})
        return data
    except Exception as e:
        st.error(f"Error loading original dataset for options: {e}")
        return pd.DataFrame(columns=['age', 'workclass', 'fnlwgt', 'education', 'educational-num', 'marital-status',
                                    'occupation', 'relationship', 'race', 'gender', 'capital-gain', 'capital-loss',
                                    'hours-per-week', 'native-country', 'income'])

original_data_for_options = load_original_data_for_options()

# --- Preprocessing for Uploaded Income Data ---
def apply_initial_preprocessing_for_upload(df, numerical_columns, categorical_columns):
    df_processed = df.copy()
    df_processed.columns = df_processed.columns.str.strip()
    for col in df_processed.select_dtypes(include=['object']).columns:
        df_processed[col] = df_processed[col].str.strip()
    all_expected_input_cols = numerical_columns + categorical_columns + ['education']
    missing_cols_in_upload = [col for col in all_expected_input_cols if col not in df_processed.columns]
    if missing_cols_in_upload:
        st.error(f"Uploaded dataset is missing essential columns: {', '.join(missing_cols_in_upload)}. Please check your file headers.")
        return None
    initial_rows = df_processed.shape[0]
    df_processed = df_processed[(df_processed['age'] <= 75) & (df_processed['age'] >= 17)]
    st.info(f"Filtered {initial_rows - df_processed.shape[0]} rows based on age (17-75). Remaining: {df_processed.shape[0]} in uploaded data.")
    initial_rows = df_processed.shape[0]
    df_processed['workclass'] = df_processed['workclass'].replace('?', 'Others')
    df_processed = df_processed[~df_processed['workclass'].isin(['Without-pay', 'Never-worked'])]
    st.info(f"Filtered {initial_rows - df_processed.shape[0]} rows based on workclass. Remaining: {df_processed.shape[0]} in uploaded data.")
    initial_rows = df_processed.shape[0]
    df_processed['education'] = df_processed['education'].replace('?', 'Others')
    df_processed = df_processed[~df_processed['education'].isin(['1st-4th', '5th-6th', 'Preschool'])]
    st.info(f"Filtered {initial_rows - df_processed.shape[0]} rows based on education. Remaining: {df_processed.shape[0]} in uploaded data.")
    for col in ['marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']:
        if col in df_processed.columns and '?' in df_processed[col].unique():
            df_processed[col] = df_processed[col].replace('?', 'Others')
    return df_processed

# --- Sidebar Model Overview ---
st.sidebar.header("🔍 Model Overview")
with st.sidebar.expander("Show Model Performance"):
    st.write("**Income Prediction (Binary)**")
    st.write("- Pipeline - Accuracy: 0.85 (estimated)")
    st.write("**Salary Prediction (Numerical)**")
    for name, (_, _) in model_options.items():
        st.write(f"**{name}**")

# --- Tabbed Interface ---
income_tab, salary_tab = st.tabs(["Income Prediction (<=50K or >50K)", "Salary Prediction (Numerical)"])

# --- Income Prediction Tab ---
with income_tab:
    st.subheader("⬆️ Upload Dataset for Batch Income Prediction")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="Upload a dataset with similar columns to 'adult 3.csv' for bulk income predictions.", key="income_uploader")
    if uploaded_file is not None:
        try:
            user_data = pd.read_csv(uploaded_file)
            st.write("Original Uploaded Data Preview:")
            st.dataframe(user_data.head())
            processed_user_data = apply_initial_preprocessing_for_upload(user_data, numerical_columns, categorical_columns)
            if processed_user_data is not None:
                X_user_for_prediction = processed_user_data.drop(columns=['income'], errors='ignore')
                st.subheader("Processing and Predicting on Your Data...")
                with st.spinner("Making predictions..."):
                    user_predictions_numeric = best_pipeline.predict(X_user_for_prediction)
                    user_predictions_proba = best_pipeline.predict_proba(X_user_for_prediction)
                user_predictions_label = np.array(['>50K' if p == 1 else '<=50K' for p in user_predictions_numeric])
                processed_user_data['Predicted_Income'] = user_predictions_label
                processed_user_data['Probability_<=50K'] = user_predictions_proba[:, 0].round(3)
                processed_user_data['Probability_>50K'] = user_predictions_proba[:, 1].round(3)
                st.subheader("Prediction Results:")
                st.dataframe(processed_user_data[['age', 'workclass', 'occupation', 'hours-per-week', 'Predicted_Income', 'Probability_<=50K', 'Probability_>50K']].head(10))
                st.write(f"Total predictions made: {len(processed_user_data)}")
                st.write("Distribution of Predicted Income:")
                st.dataframe(processed_user_data['Predicted_Income'].value_counts())
                csv_output = processed_user_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv_output,
                    file_name="income_predictions.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"An error occurred while processing your uploaded file: {e}")
            st.info("Please ensure your CSV file is correctly formatted and contains the expected columns.")

    st.markdown("---")
    
    st.subheader("🧑‍💻 Predict Income for a New Individual")
    with st.form("income_prediction_form"):
        workclass_options = sorted(original_data_for_options['workclass'].unique().tolist()) if not original_data_for_options.empty else ['Private', 'Self-emp-not-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Self-emp-inc', 'Others']
        occupation_options = sorted(original_data_for_options['occupation'].unique().tolist()) if not original_data_for_options.empty else ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Prof-specialty', 'Sales', 'Craft-repair', 'Transport-moving', 'Farming-fishing', 'Machine-op-inspct', 'Tech-support', 'Other-service', 'Protective-serv', 'Armed-Forces', 'Priv-house-serv', 'Others']
        native_country_options = sorted(original_data_for_options['native-country'].unique().tolist()) if not original_data_for_options.empty else ['United-States', 'Mexico', 'Philippines', 'Germany', 'Puerto-Rico', 'Canada', 'El-Salvador', 'India', 'Cuba', 'England', 'Jamaica', 'South', 'China', 'Italy', 'Dominican-Republic', 'Vietnam', 'Guatemala', 'Columbia', 'Poland', 'Japan', 'Iran', 'Taiwan', 'Haiti', 'Portugal', 'Nicaragua', 'Peru', 'France', 'Greece', 'Ecuador', 'Ireland', 'Hong', 'Cambodia', 'Trinadad&Tobago', 'Thailand', 'Laos', 'Yugoslavia', 'Outlying-US(Guam-USVI-etc)', 'Hungary', 'Honduras', 'Scotland', 'Holand-Netherlands', 'Others']
        marital_status_options = sorted(original_data_for_options['marital-status'].unique().tolist()) if not original_data_for_options.empty else ['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed']
        relationship_options = sorted(original_data_for_options['relationship'].unique().tolist()) if not original_data_for_options.empty else ['Husband', 'Not-in-family', 'Other-relative', 'Own-child', 'Unmarried', 'Wife']
        race_options = sorted(original_data_for_options['race'].unique().tolist()) if not original_data_for_options.empty else ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White']
        gender_options = sorted(original_data_for_options['gender'].unique().tolist()) if not original_data_for_options.empty else ['Female', 'Male']
        education_options = sorted(original_data_for_options['education'].unique().tolist()) if not original_data_for_options.empty else ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Masters', '9th', 'Doctorate', 'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Prof-school', '10th', '12th']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=17, max_value=75, value=38, help="Age of the individual (17-75)", key="income_age")
            workclass = st.selectbox("Workclass", workclass_options, help="Type of employer (e.g., Private, Self-emp-not-inc)", key="income_workclass")
            fnlwgt = st.number_input("Final Weight", min_value=10000, max_value=1500000, value=89814, help="Statistical weight (usually represents population count)", key="income_fnlwgt")
            education = st.selectbox("Education", education_options, help="Highest level of education achieved", key="income_education")
        with col2:
            educational_num = st.number_input("Educational Number", min_value=1, max_value=16, value=9, help="Numerical representation of education level (e.g., 9 for HS-grad, 13 for Bachelors)", key="income_educational_num")
            marital_status = st.selectbox("Marital Status", marital_status_options, help="Marital status (e.g., Married-civ-spouse, Never-married)", key="income_marital_status")
            occupation = st.selectbox("Occupation", occupation_options, help="Occupation of the individual", key="income_occupation")
            relationship = st.selectbox("Relationship", relationship_options, help="Relationship status (e.g., Husband, Not-in-family)", key="income_relationship")
        with col3:
            race = st.selectbox("Race", race_options, help="Race of the individual", key="income_race")
            gender = st.selectbox("Gender", gender_options, help="Gender of the individual", key="income_gender")
            capital_gain = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0, help="Capital gains from investments", key="income_capital_gain")
            capital_loss = st.number_input("Capital Loss", min_value=0, max_value=5000, value=0, help="Capital losses from investments", key="income_capital_loss")
            hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=99, value=50, help="Number of hours worked per week", key="income_hours_per_week")
            native_country = st.selectbox("Native Country", native_country_options, help="Country of origin", key="income_native_country")
        
        st.markdown("---")
        submitted = st.form_submit_button("🚀 Predict Income")
        
        if submitted:
            new_individual = {
                'age': age,
                'workclass': workclass,
                'fnlwgt': fnlwgt,
                'education': education,
                'educational-num': educational_num,
                'marital-status': marital_status,
                'occupation': occupation,
                'relationship': relationship,
                'race': race,
                'gender': gender,
                'capital-gain': capital_gain,
                'capital-loss': capital_loss,
                'hours-per-week': hours_per_week,
                'native-country': native_country
            }
            prediction_df_single = pd.DataFrame([new_individual])
            try:
                prediction = best_pipeline.predict(prediction_df_single)
                prediction_proba = best_pipeline.predict_proba(prediction_df_single)
                prediction_label = '>50K' if prediction[0] == 1 else '<=50K'
                st.success(f"**🎉 Prediction: {prediction_label}**")
                st.info(f"Probability of <=50K: {prediction_proba[0][0]:.2f}, Probability of >50K: {prediction_proba[0][1]:.2f}")
            except Exception as e:
                st.error(f"An error occurred during income prediction: {e}")

# --- Salary Prediction Tab ---
with salary_tab:
    model_choice = st.selectbox("Select a prediction model:", list(model_options.keys()), key="salary_model_choice")
    model_path, _ = model_options[model_choice]
    salary_model = load_salary_model(model_path)
    
    st.subheader("🧑‍💻 Predict Salary for a New Individual")
    with st.form("salary_prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30, help="Age of the individual (18-100)", key="salary_age")
            gender = st.selectbox("Gender", ["Male", "Female"], help="Gender of the individual", key="salary_gender")
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"], help="Marital status", key="salary_marital_status")
            job_title = st.text_input("Job Title", value="", help="Job title of the individual", key="salary_job_title")
        with col2:
            experience = st.slider("Years of Experience", 0, 40, 5, help="Years of professional experience", key="salary_experience")
            education_level = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"], help="Highest level of education", key="salary_education_level")
            education_numeric = st.slider("Education Score (0-10)", 0, 10, 5, help="Numerical education score", key="salary_education_numeric")
        with col3:
            hours_per_week = st.slider("Hours Worked per Week", 20, 80, 40, help="Average hours worked per week", key="salary_hours_per_week")
            city = st.text_input("City", value="", help="City of residence", key="salary_city")
            location_type = st.selectbox("Location Type", ["Urban", "Suburban", "Rural"], help="Type of location", key="salary_location_type")
            nationality = st.text_input("Nationality", value="", help="Nationality of the individual", key="salary_nationality")
        
        st.markdown("---")
        submitted = st.form_submit_button("🚀 Predict Salary")
        
        if submitted:
            input_data = pd.DataFrame({
                "age": [age],
                "gender": [gender],
                "marital_status": [marital_status],
                "job_title": [job_title],
                "experience": [experience],
                "education_level": [education_level],
                "education_numeric": [education_numeric],
                "hours_per_week": [hours_per_week],
                "city": [city],
                "location_type": [location_type],
                "nationality": [nationality]
            })
            try:
                # Assuming models handle categorical data internally
                prediction = salary_model.predict(input_data)[0]
                st.success(f"💰 Predicted Salary: ${prediction:,.2f}")
            except Exception as e:
                st.error(f"Prediction failed: {e}. Ensure input data matches the model's training format.")

# --- Notes Section ---
st.markdown("""
---
### 📘 Notes:
- **Income Prediction**: Predicts if income is <=50K or >50K using a pre-trained pipeline based on the UCI Adult dataset.
- **Salary Prediction**: Estimates numerical salary using various models. Ensure input structure matches the training pipeline.
- All models are pre-trained and loaded from `.pkl` files in `C:\\Users\\subod\\Downloads\\salary_prediction\\trained_models`.
- Use simpler models (e.g., Logistic Regression) for explainability, or complex models (e.g., XGBoost, Stacking) for better accuracy.

### 🔧 Tips:
- Ensure consistent preprocessing during training and inference.
- For income prediction, the dataset must match the structure of 'adult 3.csv'.
- For salary prediction, input fields should align with the model's training data. If predictions fail, verify that the model handles categorical data internally or adjust the input format.
- If files are missing, run 'train_model.py' for income prediction or the salary training script for salary models.
""")

st.markdown("---")
st.write("Developed by Subodh Swaroop Achar")
