import pandas as pd
import numpy as np
import joblib
import sys

# Optionally increase recursion limit if necessary
sys.setrecursionlimit(10000)

# Define file paths for the datasets
file_mean = "/Users/dianenacario/scikit_learn_data/uol_group_d/datasets_mean_median/LAEI_2019_NA_FILLED_WITH_MEAN.csv"
file_median = "/Users/dianenacario/scikit_learn_data/uol_group_d/datasets_mean_median/LAEI_2019_NA_FILLED_WITH_MEDIAN.csv"

# Reload the datasets
print("Loading datasets...")
train_data_mean = pd.read_csv(file_mean)
train_data_median = pd.read_csv(file_median)
print("Datasets loaded successfully!")

# Define pollutants to predict
pollutants = ["nox", "pm10", "pm2.5", "co2"]

# Load the trained models from the saved files
models_mean = {}
models_median = {}

print("Loading models...")
for pollutant in pollutants:
    models_mean[pollutant] = joblib.load(f'rf_model_mean_{pollutant}.pkl')
    models_median[pollutant] = joblib.load(f'rf_model_median_{pollutant}.pkl')
print("Models loaded successfully!")

# Ensure all categorical columns are properly encoded before training
print("One-Hot Encoding categorical columns...")

# We identify categorical columns that need encoding
categorical_cols = ['Main Source Category', 'Borough', 'Sector', 'Source', 'Emissions Unit', 'Zone']

# One-Hot Encode the categorical columns for both the training and prediction datasets
train_data_mean_encoded = pd.get_dummies(train_data_mean, columns=categorical_cols, drop_first=True)
train_data_median_encoded = pd.get_dummies(train_data_median, columns=categorical_cols, drop_first=True)

# Generate input features for 2025 using the mean and median of the historical data (2013, 2016, 2019)
print("Generating input features for 2025...")

# Select the relevant data for 2013, 2016, and 2019, excluding pollutants and Year
X_train_mean = train_data_mean[train_data_mean["Year"].isin([2013, 2016, 2019])].drop(columns=pollutants + ["Year"])
X_train_median = train_data_median[train_data_median["Year"].isin([2013, 2016, 2019])].drop(columns=pollutants + ["Year"])

# Calculate mean and median for the numeric columns for 2025 features
numeric_cols = X_train_mean.select_dtypes(include=[np.number]).columns.tolist()

# Create 2025 numeric features
X_2025_mean_numeric = X_train_mean[numeric_cols].mean().values.reshape(1, -1)
X_2025_median_numeric = X_train_median[numeric_cols].median().values.reshape(1, -1)

# Extract one-hot encoded categorical columns from 2013, 2016, 2019
categorical_encoded_cols = train_data_mean_encoded.columns.difference(numeric_cols + pollutants + ["Year"]).tolist()

# Mode is used for categorical features
X_2025_categorical = train_data_mean_encoded[categorical_encoded_cols][train_data_mean["Year"].isin([2013, 2016, 2019])].mode().iloc[0].values.reshape(1, -1)

# Combine numeric and categorical columns for 2025 prediction
X_2025_mean_encoded = pd.DataFrame(np.hstack([X_2025_mean_numeric, X_2025_categorical]), columns=numeric_cols + categorical_encoded_cols)
X_2025_median_encoded = pd.DataFrame(np.hstack([X_2025_median_numeric, X_2025_categorical]), columns=numeric_cols + categorical_encoded_cols)

# Align the columns to match the model's input columns
print("Aligning columns with the training set encoding...")

# Retrieve the feature names the model was trained on
trained_feature_names = models_mean['nox'].feature_names_in_

def align_features(X_2025, trained_feature_names):
    # Add missing features with default values (e.g., 0)
    missing_features = set(trained_feature_names) - set(X_2025.columns)
    for feature in missing_features:
        X_2025[feature] = 0
    
    # Ensure the columns are in the same order as trained features
    X_2025 = X_2025[trained_feature_names]
    return X_2025

X_2025_mean_encoded_aligned = align_features(X_2025_mean_encoded, trained_feature_names)
X_2025_median_encoded_aligned = align_features(X_2025_median_encoded, trained_feature_names)

# Predict pollutant levels for 2025 using both mean and median-imputed models
predictions_2025_mean = {}
predictions_2025_median = {}

print("Predicting pollutant levels for 2025...")
for pollutant in pollutants:
    # Predict using the mean-imputed model
    predictions_2025_mean[pollutant] = models_mean[pollutant].predict(X_2025_mean_encoded_aligned)
    
    # Predict using the median-imputed model
    predictions_2025_median[pollutant] = models_median[pollutant].predict(X_2025_median_encoded_aligned)

# Display predictions for each pollutant
print("Displaying predictions for 2025...")
for pollutant in pollutants:
    print(f"Predicted {pollutant} for 2025 (Mean Imputed): {predictions_2025_mean[pollutant][0]}")
    print(f"Predicted {pollutant} for 2025 (Median Imputed): {predictions_2025_median[pollutant][0]}")

print("Prediction completed successfully!")
## FINAL OUTPUT ##
# Predicting pollutant levels for 2025...
# Displaying predictions for 2025...
# Predicted nox for 2025 (Mean Imputed): 4.487611659936521
# Predicted nox for 2025 (Median Imputed): 3.50199304871478e-11
# Predicted pm10 for 2025 (Mean Imputed): 0.07071947975409391
# Predicted pm10 for 2025 (Median Imputed): 6.566573086596422e-10
# Predicted pm2.5 for 2025 (Mean Imputed): 0.13327198295378076
# Predicted pm2.5 for 2025 (Median Imputed): 1.759428378883236e-07
# Predicted co2 for 2025 (Mean Imputed): 1814.5586204028068
# Predicted co2 for 2025 (Median Imputed): 3.838912905694337e-11
# Prediction completed successfully!