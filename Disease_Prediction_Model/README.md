# Disease Prediction from Medical Data

A comprehensive machine learning pipeline for predicting diseases from patient medical data using multiple classification algorithms.

## Project Overview

This project implements a robust disease prediction system that can classify patients into disease risk categories using various machine learning algorithms. The system supports multiple disease types and provides comprehensive evaluation metrics.

## Features

- **Multiple Disease Support**: Heart Disease, Diabetes, Breast Cancer
- **Advanced ML Algorithms**: Random Forest, SVM, Logistic Regression
- **Comprehensive Evaluation**: ROC AUC, F1-Score, Precision, Recall, Confusion Matrix
- **Data Preprocessing**: Automatic handling of missing values, categorical encoding, scaling
- **Imbalanced Learning**: SMOTE for handling class imbalance
- **Model Persistence**: Save and load trained models
- **Professional Output**: Clean metrics, visualizations, and predictions

## Model Performance

| Disease | Best Model | Accuracy | ROC AUC | F1-Score |
|---------|------------|----------|---------|----------|
| Heart Disease | Random Forest | 100.0% | 100.0% | 100.0% |
| Diabetes | SVM | 74.0% | 82.6% | 65.5% |
| Breast Cancer | Random Forest | 100.0% | 100.0% | 100.0% |

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd disease-prediction

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Train Models

```bash
# Train heart disease model
python disease_prediction_pipeline.py --data heart.csv --target target --outdir runs/heart --smote --calibrate

# Train diabetes model
python disease_prediction_pipeline.py --data diabetes.csv --target Outcome --outdir runs/diabetes --smote

# Train breast cancer model
python disease_prediction_pipeline.py --data breast_cancer.csv --target target --outdir runs/breast_cancer --smote --calibrate
```

### 2. Make Predictions

```bash
# Predict for new patients
python predict_new_patients.py
```

### 3. Run Tests

```bash
# Test the pipeline
python test_pipeline.py
```

## Project Structure

```
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── disease_prediction_pipeline.py # Main ML pipeline
├── predict_new_patients.py      # Prediction script
├── example_usage.py             # Usage examples
├── test_pipeline.py             # Test suite
├── heart.csv                    # Heart disease dataset
├── diabetes.csv                 # Diabetes dataset
├── breast_cancer.csv            # Breast cancer dataset
├── runs/                        # Model outputs and results
│   ├── heart/                   # Heart disease results
│   ├── diabetes/                # Diabetes results
│   └── breast_cancer/           # Breast cancer results
└── .gitignore                   # Git ignore file
```

## Usage Examples

### Basic Usage

```python
from disease_prediction_pipeline import main

# Train heart disease model
main(['--data', 'heart.csv', '--target', 'target', '--outdir', 'runs/heart', '--smote'])
```

### Making Predictions

```python
import joblib
import pandas as pd

# Load trained model
model = joblib.load('runs/heart/rf_best_model.pkl')

# Prepare patient data
patient_data = pd.DataFrame({
    'age': [45],
    'sex': [1],
    'cp': [0],
    'trestbps': [130],
    'chol': [250],
    'fbs': [0],
    'restecg': [0],
    'thalach': [150],
    'exang': [0],
    'oldpeak': [0.0],
    'slope': [2],
    'ca': [0],
    'thal': [2]
})

# Make prediction
prediction = model.predict(patient_data)
probability = model.predict_proba(patient_data)[0][1]
print(f"Prediction: {'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'}")
print(f"Probability: {probability:.1%}")
```

## Datasets

### Heart Disease Dataset
- **Features**: Age, Sex, Chest Pain Type, Blood Pressure, Cholesterol, etc.
- **Target**: Heart disease presence (0/1)
- **Samples**: 1,027 patients

### Diabetes Dataset
- **Features**: Pregnancies, Glucose, Blood Pressure, BMI, Age, etc.
- **Target**: Diabetes diagnosis (0/1)
- **Samples**: 769 patients

### Breast Cancer Dataset
- **Features**: Tumor radius, texture, perimeter, area, smoothness, etc.
- **Target**: Malignant/Benign (0/1)
- **Samples**: 569 patients

## Key Features

### Machine Learning Pipeline
- **Data Preprocessing**: Automatic cleaning, encoding, scaling
- **Feature Engineering**: Permutation importance analysis
- **Model Selection**: Cross-validation with multiple algorithms
- **Evaluation**: Comprehensive metrics and visualizations
- **Model Persistence**: Save/load trained models

### Advanced Techniques
- **SMOTE**: Handle imbalanced datasets
- **Probability Calibration**: Well-calibrated probability estimates
- **Feature Importance**: Identify key medical indicators
- **Cross-Validation**: Robust model evaluation

## Results and Visualizations

The pipeline generates:
- **Performance Metrics**: Accuracy, ROC AUC, F1-Score, Precision, Recall
- **Visualizations**: ROC curves, PR curves, confusion matrices
- **Feature Importance**: Key medical indicators for each disease
- **Model Comparisons**: Performance across different algorithms

## Testing

```bash
# Run comprehensive tests
python test_pipeline.py
```

Tests include:
- Data loading and preprocessing
- Model training and evaluation
- Prediction functionality
- Package availability checks

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- UCI Machine Learning Repository for datasets
- Scikit-learn for ML algorithms
- Medical professionals for domain expertise

## Contact

For questions or contributions, please open an issue on GitHub.

---

**Built with care for healthcare applications**