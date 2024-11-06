import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(train_path, test_path):
    """
    Load and preprocess the training and test datasets using only RF feature importance
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
    
    # Feature selection using Random Forest importance only
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_selector.fit(X_train, y_train)
    
    # Calculate and save feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': rf_selector.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Save feature importance analysis
    feature_importance.to_csv('feature_importance_analysis.csv', index=False)
    
    # Select features based on importance threshold
    selected_features = feature_importance[
        feature_importance['Importance'] > 0.05
    ]['Feature'].tolist()
    
    # Prepare final datasets with selected features
    train_processed = all_data[:len(train_df)][selected_features]
    test_processed = all_data[len(train_df):][selected_features]
    
    return train_processed, test_processed, y_train, selected_features

def train_model(X_train, y_train):
    """
    Train the Random Forest model
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,          # Added to control model complexity
        min_samples_split=5,   # Added to prevent overfitting
        min_samples_leaf=2,    # Added to prevent overfitting
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_train, y_train, selected_features):
    """
    Evaluate model performance and create visualization plots
    """
    # Get predictions and probabilities
    y_pred = model.predict(X_train)
    y_prob = model.predict_proba(X_train)[:, 1]
    
    # Calculate and plot confusion matrix
    cm = confusion_matrix(y_train, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Calculate and plot ROC curve
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
        'Feature': selected_features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance (Selected Features)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    # Generate classification report
    report = classification_report(y_train, y_pred)
    with open('model_evaluation.txt', 'w') as f:
        f.write(f"Selected Features: {', '.join(selected_features)}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write(f"\nROC AUC Score: {roc_auc:.4f}")
    
    return report, roc_auc

def main():
    # Load and preprocess data with simplified feature selection
    X_train, X_test, y_train, selected_features = load_and_preprocess_data(
        'train.csv',
        'test.csv'
    )
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Make predictions on test set
    test_prob = model.predict_proba(X_test)[:, 1]
    
    # Save predictions to CSV
    predictions_df = pd.DataFrame({
        'Predicted_Probability': test_prob
    })
    predictions_df.to_csv('predictions.csv', index=False)
    
    # Evaluate model and save results
    report, roc_auc = evaluate_model(model, X_train, y_train, selected_features)
    
    # Print summary
    print("\nModel Training Complete!")
    print(f"Number of selected features: {len(selected_features)}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print("\nFiles generated:")
    print("- predictions.csv (Predicted probabilities)")
    print("- feature_importance_analysis.csv (Feature importance details)")
    print("- model_evaluation.txt (Performance metrics)")
    print("- confusion_matrix.png")
    print("- roc_curve.png")
    print("- feature_importance.png")

if __name__ == "__main__":
    main()