# Required Libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer

# Load Data
def load_data(dev_path, val_path):
    if not os.path.exists(dev_path):
        raise FileNotFoundError(f"Development data file not found: {dev_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation data file not found: {val_path}")

    dev_data = pd.read_csv(dev_path)
    val_data = pd.read_csv(val_path)
    return dev_data, val_data

# Exploratory Data Analysis (EDA)
def perform_eda(dev_data):
    print("EDA Summary")
    print(dev_data.info())
    print(dev_data.describe())
    print(dev_data["bad_flag"].value_counts(normalize=True))

    # Check for missing values
    missing = dev_data.isnull().sum()
    print("Missing Values:\n", missing[missing > 0])

# Preprocessing
def preprocess_data(dev_data):
    # Data cleaning: Handle invalid or inconsistent values
    for column in dev_data.columns:
        if dev_data[column].dtype == 'object':
            dev_data[column] = dev_data[column].str.strip().replace({"": np.nan})

    # Drop columns with all NaN values before imputation
    dev_data = dev_data.dropna(axis=1, how='all')

    # Handle missing values explicitly
    imputer = SimpleImputer(strategy='median')
    numeric_columns = dev_data.select_dtypes(include=[np.number]).columns
    dev_data.loc[:, numeric_columns] = imputer.fit_transform(dev_data[numeric_columns])

    # Check for remaining NaNs (should be none)
    if dev_data.isnull().any().any():
        raise ValueError("Preprocessing step failed to remove all NaN values.")

    # Feature-target split
    X = dev_data.drop(["account_number", "bad_flag"], axis=1)
    y = dev_data["bad_flag"]

    # Normalize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )

    return X_scaled, y, scaler

# Train-Test Split
def split_data(X, y):
    return train_test_split(X, y, test_size=0.3, random_state=42)

# Model Training
def train_model(X_train, y_train):
    gbc = GradientBoostingClassifier()
    params = {
        'n_estimators': [50, 100],  # Reduced parameter options
        'learning_rate': [0.1],    # Fixed learning rate for simplicity
        'max_depth': [3, 5]        # Smaller range
    }
    grid = GridSearchCV(gbc, params, scoring='roc_auc', cv=3, n_jobs=-1)  # Parallel processing

    try:
        # Optionally use a smaller subset for testing
        X_train_sample, _, y_train_sample, _ = train_test_split(X_train, y_train, train_size=0.2, random_state=42)
        grid.fit(X_train_sample, y_train_sample)
        print("Best Parameters:", grid.best_params_)
        return grid.best_estimator_
    except ValueError as e:
        print("GridSearchCV failed. Error details:", e)
        return None

# Evaluation
def evaluate_model(model, X_test, y_test):
    try:
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_prob)
        print("AUC-ROC Score:", auc_score)

        # Model accuracy metrics
        y_pred = model.predict(X_test)
        print("Classification Report:\n", classification_report(y_test, y_pred))
    except NotFittedError as e:
        print("Model evaluation failed. Ensure the model is trained properly.", e)

# Prediction on Validation Data
def predict_validation(model, val_data, scaler):
    # Data cleaning and missing value handling for validation data
    for column in val_data.columns:
        if val_data[column].dtype == 'object':
            val_data[column] = val_data[column].str.strip().replace({"": np.nan})

    # Drop columns with all NaN values before imputation
    val_data = val_data.dropna(axis=1, how='all')

    imputer = SimpleImputer(strategy='median')
    numeric_columns = val_data.select_dtypes(include=[np.number]).columns
    val_data.loc[:, numeric_columns] = imputer.fit_transform(val_data[numeric_columns])

    val_features = val_data.drop(["account_number"], axis=1)
    val_scaled = pd.DataFrame(
        scaler.transform(val_features),
        columns=val_features.columns
    )

    try:
        predictions = model.predict_proba(val_scaled)[:, 1]
        val_data["predicted_probability"] = predictions

        # Export predictions to CSV
        submission = val_data[["account_number", "predicted_probability"]]
        submission.to_csv("predictions.csv", index=False)
        print("Predictions exported to 'predictions.csv'")
    except NotFittedError as e:
        print("Prediction failed. Ensure the model is trained properly.", e)

# Execution
try:
    # File paths
    dev_path = "Dev_data_to_be_shared.csv"
    val_path = "Validation_data_to_be_shared.csv"

    # Load data
    dev_data, val_data = load_data(dev_path, val_path)

    # EDA
    perform_eda(dev_data)

    # Preprocess data
    X, y, scaler = preprocess_data(dev_data)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train model
    model = train_model(X_train, y_train)
    if model is not None:
        # Evaluate model
        evaluate_model(model, X_test, y_test)

        # Predict on validation data
        predict_validation(model, val_data, scaler)
    else:
        print("Model training was unsuccessful. Exiting pipeline.")

except Exception as e:
    print(f"Pipeline execution failed. Error details: {e}")
