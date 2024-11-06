import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess the training and test datasets
def load_and_preprocess_data(train_path, test_path):

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

# Train a Random Forest model
def train_model(X_train, y_train):

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_train, y_train):
    """
    Evaluate model performance and create visualization plots
    """
    # Get predictions and probabilities
    y_pred = model.predict(X_train)
    y_prob = model.predict_proba(X_train)[:, 1]
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_train, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_train, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()
    
    # Plot feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    # Generate and save classification report
    report = classification_report(y_train, y_pred)
    with open('model_evaluation.txt', 'w') as f:
        f.write(report)
    
    return report