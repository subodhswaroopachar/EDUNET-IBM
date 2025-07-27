import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import json
import os
from sklearn import set_config
set_config(transform_output="pandas")


# --- Custom Preprocessor (used by all models) ---
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

        for col in ['workclass', 'occupation', 'native-country']:
            if col in X_transformed.columns:
                X_transformed[col] = X_transformed[col].replace('?', 'Others')

        if 'education' in X_transformed.columns:
            X_transformed = X_transformed.drop(columns=['education'], errors='ignore')

        for col, le in self.label_encoders.items():
            if col in X_transformed.columns:
                unseen_mask = ~X_transformed[col].isin(le.classes_)
                X_transformed.loc[unseen_mask, col] = 'Others'
                X_transformed[col] = le.transform(X_transformed[col].astype(str))

        return pd.DataFrame(X_transformed, columns=X_transformed.columns)


# --- Load and preprocess data ---
def load_and_preprocess_data(csv_file_path):
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"File not found: {csv_file_path}")

    data = pd.read_csv(csv_file_path)
    data.columns = data.columns.str.strip()
    for col in data.select_dtypes(include='object').columns:
        data[col] = data[col].str.strip()

    # Filter and clean
    data = data[(data['age'] >= 17) & (data['age'] <= 75)]
    data['workclass'] = data['workclass'].replace('?', 'Others')
    data = data[~data['workclass'].isin(['Without-pay', 'Never-worked'])]
    data['education'] = data['education'].replace('?', 'Others')
    data = data[~data['education'].isin(['1st-4th', '5th-6th', 'Preschool'])]

    for col in ['marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']:
        if col in data.columns:
            data[col] = data[col].replace('?', 'Others')

    if 'income' not in data.columns:
        raise ValueError("'income' column missing after preprocessing.")

    data['income'] = data['income'].replace({'>50K.': '>50K', '<=50K.': '<=50K'})
    X = data.drop(columns='income')
    y = data['income'].apply(lambda x: 1 if x == '>50K' else 0)

    numerical_columns = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    categorical_columns = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']

    missing = [col for col in numerical_columns + categorical_columns + ['education'] if col not in X.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return X, y, numerical_columns, categorical_columns


# --- Main Training Script ---
if __name__ == "__main__":
    csv_path = r"C:\Users\subod\Downloads\adult 3.csv"
    model_dir = "trained_models"
    os.makedirs(model_dir, exist_ok=True)

    try:
        X, y, num_cols, cat_cols = load_and_preprocess_data(csv_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    preprocessor = ColumnTransformer([
        ('num', MinMaxScaler(), num_cols),
        ('cat', 'passthrough', [c for c in cat_cols if c in X.columns])
    ])

    models = [
        ('KNN', KNeighborsClassifier()),
        ('Decision Tree', DecisionTreeClassifier(random_state=42)),
        ('Random Forest', RandomForestClassifier(random_state=42)),
        ('Extra Trees', ExtraTreesClassifier(random_state=42)),
        ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
        ('Naive Bayes', GaussianNB()),
        ('MLP Classifier', MLPClassifier(max_iter=1000, random_state=42)),
        ('XGBoost', XGBClassifier(random_state=42, eval_metric='logloss')),
        ('LightGBM', LGBMClassifier(random_state=42, verbose=-1)),
        ('CatBoost', CatBoostClassifier(random_state=42, verbose=0)),
        ('HistGradientBoosting', HistGradientBoostingClassifier(random_state=42)),
        ('Stacking', StackingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(random_state=42)),
                ('xgb', XGBClassifier(random_state=42, eval_metric='logloss')),
                ('lgbm', LGBMClassifier(random_state=42, verbose=-1)),
            ],
            final_estimator=LogisticRegression(max_iter=1000, random_state=42)
        ))
    ]

    best_score = -1
    best_model = None
    best_name = ""

    for name, model in models:
        print(f"\n🔧 Training: {name}")
        pipeline = Pipeline([
            ('custom_preprocessor', CustomPreprocessor()),
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        try:
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
            acc = accuracy_score(y_test, preds)
            print(f"✅ Accuracy: {acc:.4f}")

            # Save each model pipeline
            save_name = name.lower().replace(" ", "_").replace("-", "") + "_pipeline.pkl"
            joblib.dump(pipeline, os.path.join(model_dir, save_name))

            if acc > best_score:
                best_score = acc
                best_model = pipeline
                best_name = name

        except Exception as e:
            print(f"❌ Error training {name}: {e}")

    if best_model:
        print(f"\n🏆 Best Model: {best_name} | Accuracy: {best_score:.4f}")
        joblib.dump(best_model, os.path.join(model_dir, "best_income_prediction_pipeline.pkl"))

        with open(os.path.join(model_dir, 'numerical_columns.json'), 'w') as f:
            json.dump(num_cols, f)
        with open(os.path.join(model_dir, 'categorical_columns.json'), 'w') as f:
            json.dump(cat_cols, f)

        print(f"📦 Models and metadata saved to '{model_dir}'")
    else:
        print("❗ No model was successfully trained.")
