# Customer Churn Prediction Model

A machine learning pipeline for predicting customer churn using Random Forest classifier with consistent parameter settings throughout the feature selection and model training process.

## Project Structure

```
.
├── main.py                           # Main script containing the ML pipeline
├── train.csv                         # Training dataset
├── test.csv                          # Test dataset
├── predictions.csv                   # Binary predictions and probabilities
├── feature_importance_analysis.csv   # Feature importance analysis
├── model_evaluation.txt              # Classification report and metrics
├── confusion_matrix.png             # Confusion matrix visualization
├── roc_curve.png                    # ROC curve visualization
├── feature_importance.png           # Feature importance plot
└── README.md                        # This file
```

## Features

### Model Parameters
The implementation uses consistent Random Forest parameters throughout the pipeline:
```python
RF_PARAMS = {
    'n_estimators': 100,    # Number of trees
    'max_depth': 10,        # Maximum tree depth
    'min_samples_split': 5, # Minimum samples for node splitting
    'min_samples_leaf': 2,  # Minimum samples in leaf nodes
    'random_state': 42      # For reproducibility
}
```

### Data Preprocessing
- Label encoding for categorical variables
- Standard scaling for numerical features
- Consistent preprocessing across training and test sets

### Feature Selection
- Uses Random Forest importance scores with consistent parameters
- Importance threshold: > 0.05 (5%)
- Features selected based on importance analysis

## Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Input Data Format

### Required Columns
1. ID Columns:
   - 'id'
   - 'CustomerId'

2. Categorical Features:
   - 'Geography'
   - 'Gender'
   - 'Surname'

3. Numerical Features:
   - 'CreditScore'
   - 'Age'
   - 'Tenure'
   - 'Balance'
   - 'NumOfProducts'
   - 'HasCrCard'
   - 'IsActiveMember'
   - 'EstimatedSalary'

4. Target Variable (training data only):
   - 'Exited'

## Usage

1. Prepare your data files:
   - Place `train.csv` and `test.csv` in the same directory as `main.py`
   - Ensure all required columns are present

2. Run the pipeline:
```bash
python main.py
```

## Output Files

1. `predictions.csv`
   - Contains two columns:
     - `Predicted_Exited`: Binary predictions (0 or 1)
     - `Prediction_Probability`: Probability scores

2. `feature_importance_analysis.csv`
   - Complete feature importance scores
   - Sorted by importance value

3. `model_evaluation.txt`
   - List of selected features
   - Classification report
   - ROC AUC score

4. Visualizations:
   - `confusion_matrix.png`: Model prediction accuracy
   - `roc_curve.png`: ROC curve with AUC score
   - `feature_importance.png`: Importance of selected features

## Model Evaluation

The pipeline generates:
1. Classification Metrics:
   - Precision
   - Recall
   - F1-score
   - Support

2. Visualizations:
   - Confusion Matrix
   - ROC Curve with AUC score
   - Feature Importance Plot

## Error Handling

The pipeline includes checks for:
1. Data Loading:
   - Presence of required columns
   - Proper file formats

2. Preprocessing:
   - Categorical encoding
   - Numerical scaling
   - Feature selection validation

## Customization

1. Modify model parameters in `main.py`:
```python
RF_PARAMS = {
    'n_estimators': your_value,
    'max_depth': your_value,
    'min_samples_split': your_value,
    'min_samples_leaf': your_value,
    'random_state': 42
}
```

2. Adjust feature selection threshold:
```python
feature_importance['Importance'] > your_threshold  # Default is 0.05
```

## Troubleshooting

Common issues and solutions:

1. Missing Columns:
   - Verify all required columns exist in input files
   - Check column names match exactly

2. Memory Issues:
   - Reduce n_estimators in RF_PARAMS
   - Increase min_samples_split threshold

3. Performance Issues:
   - Adjust feature importance threshold
   - Modify model parameters

## Future Improvements

Potential enhancements:
1. Add cross-validation
2. Implement grid search for parameter optimization
3. Add feature correlation analysis
4. Include more performance metrics
