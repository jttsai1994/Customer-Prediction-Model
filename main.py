import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(train_path, test_path):
    """
    Load and preprocess the training and test datasets
    """
    # Load datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Define column types
    id_columns = ['id', 'CustomerId']
    categorical_cols = ['Geography', 'Gender', 'Surname']
    numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 
                     'NumOfProducts', 'HasCrCard', 'IsActiveMember', 
                     'EstimatedSalary']
    
    # Combine train and test for consistent encoding
    all_data = pd.concat([train_df, test_df], axis=0)
    
    # Handle categorical variables
    le = LabelEncoder()
    for col in categorical_cols:
        if col in all_data.columns:
            all_data[col] = le.fit_transform(all_data[col])
    
    # Scale numerical features
    scaler = StandardScaler()
    all_data[numerical_cols] = scaler.fit_transform(all_data[numerical_cols])
    
    # Prepare data for feature selection
    X_train = all_data[:len(train_df)].drop(id_columns + ['Exited'], axis=1)
    y_train = train_df['Exited']
    
    # Feature selection using Random Forest importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Select features based on importance threshold
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    selected_features = importance_df[importance_df['Importance'] > 0.05]['Feature'].tolist()
    
    # Prepare final datasets with selected features
    train_processed = all_data[:len(train_df)][selected_features]
    test_processed = all_data[len(train_df):][selected_features]
    
    return train_processed, test_processed, y_train