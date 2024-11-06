# Customer Churn Prediction Model

A machine learning pipeline for predicting customer churn using Random Forest with consistent parameter settings and robust feature selection.

## Project Structure

```
.
├── main.py                           # Main script with ML pipeline
├── train_example.csv                 # Training dataset
├── test.csv                          # Test dataset
├── predictions.csv                   # Model predictions
├── initial_feature_importance.csv    # Initial feature importance analysis
├── feature_importance_comparison.csv # Parameter impact comparison
├── importance_comparison.png        # Visual comparison of importance scores
└── README.md                        # This file
```

## Features

### Consistent Model Parameters
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

### Feature Selection Process
- Uses Random Forest importance scores
- Importance threshold: > 0.05 (5%)
- Same parameters for both selection and final model
- Comparison analysis of feature importance stability

### Data Preprocessing
- Label encoding for categorical variables
- Standard scaling for numerical features
- Consistent preprocessing across train and test sets

## Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Input Data Format

### Training Data Columns
- ID Columns: 'id', 'CustomerId'
- Categorical: 'Geography', 'Gender', 'Surname'
- Numerical: 
  - 'CreditScore'
  - 'Age'
  - 'Tenure'
  - 'Balance'
  - 'NumOfProducts'
  - 'HasCrCard'
  - 'IsActiveMember'
  - 'EstimatedSalary'
- Target: 'Exited'

### Test Data
Same format as training data without 'Exited' column

## Usage

1. Prepare your data files:
   - Ensure `train_example.csv` and `test.csv` are in the correct format
   - Place them in the same directory as `main.py`

2. Run the pipeline:
```bash
python main.py
```

## Output Files

1. `predictions.csv`
   - Contains predicted churn probabilities for test set
   - Format: single column of probabilities

2. `initial_feature_importance.csv`
   - Features and their importance scores
   - Sorted by importance

3. `feature_importance_comparison.csv`
   - Comparison of feature importance with different parameters
   - Shows stability of feature selection

4. `importance_comparison.png`
   - Visual representation of feature importance comparison
   - Helps identify any parameter impact on feature selection

## Model Details

### Parameters Selection Rationale
- `max_depth=10`: Prevents overfitting while maintaining predictive power
- `min_samples_split=5`: Ensures robust node splitting
- `min_samples_leaf=2`: Maintains prediction stability
- These parameters are used consistently in both feature selection and final model

### Feature Selection Threshold
- 5% importance threshold
- Features contributing less than 5% to decisions are excluded
- Balances between model complexity and performance

## Performance Metrics
The pipeline generates several performance metrics:
- ROC curve and AUC score
- Confusion matrix
- Classification report (precision, recall, F1-score)
- Feature importance stability analysis

## Error Handling and Validation
- Consistent parameter usage throughout pipeline
- Feature importance comparison for validation
- Robust preprocessing steps
- Parameter impact analysis

## Notes
- Feature selection process uses the same model parameters as the final model to ensure consistency
- The 5% importance threshold can be adjusted based on specific needs
- Cross-validation can be added for more robust evaluation

## Customization

To modify model parameters, update `RF_PARAMS` in `main.py`:
```python
RF_PARAMS = {
    'n_estimators': your_value,
    'max_depth': your_value,
    'min_samples_split': your_value,
    'min_samples_leaf': your_value,
    'random_state': 42
}
```

To adjust feature selection threshold, modify in `load_and_preprocess_data()`:
```python
feature_importance['Importance'] > your_threshold  # Default is 0.05
```

## Troubleshooting

Common issues and solutions:
1. Missing columns in input data
   - Ensure all required columns are present
   - Check column names match exactly

2. Memory issues with large datasets
   - Reduce n_estimators in RF_PARAMS
   - Increase min_samples_split threshold

3. Feature selection too strict/loose
   - Adjust importance threshold (default 0.05)
   - Monitor number of selected features

## Future Improvements

Potential enhancements:
1. Add cross-validation for parameter tuning
2. Implement feature selection stability analysis
3. Add support for different model types
4. Include feature correlation analysis

## Contributing

To contribute:
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Submit a pull request
