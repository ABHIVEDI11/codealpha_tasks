# Disease Prediction Model

A comprehensive machine learning system for predicting multiple diseases from medical data.

## Features

- **Multiple Disease Support**: Heart Disease, Diabetes, Breast Cancer
- **Multiple Algorithms**: Logistic Regression, Random Forest, SVM
- **Synthetic Data Generation**: For demonstration and testing
- **Comprehensive Evaluation**: ROC-AUC, Precision, Recall, F1-score
- **Interactive and Batch Prediction**: Both individual and bulk processing
- **Feature Importance Analysis**: Understanding model decisions

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Models

```bash
python train.py
```

This will:
- Generate synthetic datasets for heart disease, diabetes, and breast cancer
- Train multiple models (Logistic Regression, Random Forest, SVM) for each disease
- Evaluate performance and save the best model for each disease
- Create visualization plots in the `runs/` directory

### Making Predictions

```bash
python predict.py
```

Choose from the following options:
1. **Heart Disease Prediction**: Interactive input for heart disease risk factors
2. **Diabetes Prediction**: Interactive input for diabetes risk factors
3. **Breast Cancer Prediction**: Interactive input for breast cancer features
4. **Batch Prediction**: Process multiple records from CSV files

## Disease-Specific Features

### Heart Disease
- Age, Sex, Chest Pain Type
- Blood Pressure, Cholesterol, Blood Sugar
- ECG Results, Heart Rate, Exercise Angina
- ST Depression, Vessel Count, Thalassemia

### Diabetes
- Pregnancies, Glucose Level
- Blood Pressure, Skin Thickness, Insulin
- BMI, Diabetes Pedigree, Age

### Breast Cancer
- Radius, Texture, Perimeter, Area
- Smoothness, Compactness, Concavity
- Concave Points, Symmetry, Fractal Dimension

## Output Files

### Model Files
- `heart_model.pkl`: Trained heart disease model
- `diabetes_model.pkl`: Trained diabetes model
- `breast_cancer_model.pkl`: Trained breast cancer model
- `disease_models_info.pkl`: Combined model information

### Results
- `runs/heart/disease_model_results.png`: Heart disease analysis
- `runs/diabetes/disease_model_results.png`: Diabetes analysis
- `runs/breast_cancer/disease_model_results.png`: Breast cancer analysis
- `*_predictions.csv`: Batch prediction results

## Model Performance

Each disease model provides:
- **ROC-AUC scores** for model comparison
- **Classification reports** with precision, recall, and F1-score
- **Feature importance** analysis (Random Forest)
- **Confusion matrix** visualization
- **ROC curves** for all models

## Batch Prediction

For batch processing, prepare CSV files with the required features:

### Heart Disease CSV
```csv
age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal
45,1,2,140,250,0,1,150,0,1.5,1,2,1
```

### Diabetes CSV
```csv
pregnancies,glucose,blood_pressure,skin_thickness,insulin,bmi,diabetes_pedigree,age
1,85,66,29,0,26.6,0.351,31
```

### Breast Cancer CSV
```csv
radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean
17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871
```

## Medical Disclaimer

⚠️ **Important**: This is a demonstration model using synthetic data. It should NOT be used for actual medical diagnosis. Always consult healthcare professionals for medical decisions.
