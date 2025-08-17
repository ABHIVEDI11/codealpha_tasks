# Credit Scoring Model

A machine learning model to predict creditworthiness based on financial and personal data.

## Features

- **Logistic Regression** and **Random Forest** models
- Synthetic data generation for demonstration
- Comprehensive evaluation metrics (Precision, Recall, F1-score, ROC-AUC)
- Feature importance analysis
- Interactive and batch prediction capabilities

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

```bash
python train.py
```

This will:
- Generate synthetic credit data
- Train Logistic Regression and Random Forest models
- Evaluate performance and save the best model as `credit_model.pkl`
- Create visualization plots in the `result/` directory

### Making Predictions

```bash
python predict.py
```

Choose from two options:
1. **Interactive prediction**: Enter individual customer data
2. **Batch prediction**: Process multiple records from a CSV file

### Required Features

The model expects the following features:
- `income`: Annual income
- `debt`: Total debt
- `payment_history`: Payment history score (0-100)
- `credit_utilization`: Credit utilization ratio (0-1)
- `age`: Age of applicant
- `employment_length`: Years of employment

## Output

- `credit_model.pkl`: Trained model file
- `result/credit_model_results.png`: Visualization plots
- `credit_predictions.csv`: Batch prediction results (if using batch mode)

## Model Performance

The model provides:
- ROC-AUC scores for model comparison
- Classification reports with precision, recall, and F1-score
- Feature importance analysis
- Confusion matrix visualization
