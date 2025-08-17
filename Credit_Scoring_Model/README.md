# Credit Scoring Model

A minimalistic credit scoring model that predicts an individual's creditworthiness using machine learning algorithms.

## Features

- **Multiple Models**: Logistic Regression, Decision Trees, and Random Forest
- **Feature Engineering**: Creates interaction features and risk scores
- **Comprehensive Evaluation**: Precision, Recall, F1-Score, ROC-AUC metrics
- **Visualizations**: ROC curves, confusion matrices, and feature importance plots
- **Results Export**: All results saved to organized files

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the credit scoring model:
```bash
python credit_scoring_model.py
```

## Output

The model generates a `result` folder containing:
- `model_results.txt` - Detailed performance metrics
- `model_summary.csv` - Summary table of all models
- `roc_curves.png` - ROC curve comparison
- `confusion_matrices.png` - Confusion matrices for all models
- `feature_importance.png` - Feature importance from Random Forest

## Data Features

The model uses synthetic data with the following features:
- Age, Income, Debt-to-Income ratio
- Payment history, Credit utilization
- Length of credit, Number of accounts
- Late payments (30d, 60d), Credit inquiries
- Engineered features: income-debt ratios, risk scores, age categories

## Model Performance

The model evaluates three algorithms and provides:
- AUC scores for model comparison
- Cross-validation accuracy
- Detailed classification reports
- Visual performance comparisons
