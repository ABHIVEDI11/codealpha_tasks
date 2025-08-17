#!/usr/bin/env python3
"""
Predict Disease for New Patients

This script demonstrates how to use trained models to predict
disease probability for new patient data. It includes functions
for all three disease types: Heart Disease, Diabetes, and Breast Cancer.
"""

import joblib
import pandas as pd
import numpy as np
import os
import sys

# =============================================================================
# MODEL LOADING UTILITY FUNCTION
# =============================================================================
# This function safely loads a trained machine learning model from disk:
# 1. Attempts to load the model using joblib
# 2. Provides clear success/error messages
# 3. Returns the loaded model or None if loading fails
def load_model(model_path):
    """Load a trained model from pickle file."""
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# =============================================================================
# HEART DISEASE PREDICTION FUNCTION
# =============================================================================
# This function predicts heart disease for new patients by:
# 1. Reading the best model path from a text file (or using fallback)
# 2. Loading the trained Random Forest model
# 3. Creating sample patient data with heart disease features
# 4. Making predictions and calculating probability scores
# 5. Displaying results in a clear format
def predict_heart_disease():
    """Example: Predict heart disease for new patients."""
    
    # Read the best model path from the text file
    # This allows the system to use the best performing model automatically
    try:
        with open("runs/heart/best_model.txt", "r") as f:
            model_path = f.read().strip()
    except FileNotFoundError:
        model_path = "runs/heart/rf_best_model.pkl"  # fallback to RF model
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please run the training pipeline first:")
        print("python disease_prediction_pipeline.py --data heart.csv --target target --outdir runs/heart")
        return
    
    # Load the trained model
    model = load_model(model_path)
    if model is None:
        return
    
    # Create example new patients with realistic medical data
    # Features include: age, sex, chest pain type, blood pressure, cholesterol, etc.
    new_patients = pd.DataFrame([
        {
            'age': 55, 'sex': 1, 'cp': 0, 'trestbps': 130, 'chol': 250,
            'fbs': 0, 'restecg': 1, 'thalach': 150, 'exang': 0,
            'oldpeak': 1.5, 'slope': 1, 'ca': 0, 'thal': 2
        },
        {
            'age': 65, 'sex': 0, 'cp': 2, 'trestbps': 160, 'chol': 300,
            'fbs': 1, 'restecg': 0, 'thalach': 120, 'exang': 1,
            'oldpeak': 2.5, 'slope': 0, 'ca': 2, 'thal': 3
        }
    ])
    
    print("\nHeart Disease Prediction")
    print("-" * 40)
    
    try:
        # Make predictions and get probability scores
        predictions = model.predict(new_patients)
        probabilities = model.predict_proba(new_patients)[:, 1]
        
        # Display results for each patient
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            status = "No Heart Disease" if pred == 0 else "Heart Disease"
            print(f"Patient {i+1}: {status} (Probability: {prob:.2%})")
            
    except Exception as e:
        print(f"Error making predictions: {e}")

# =============================================================================
# DIABETES PREDICTION FUNCTION
# =============================================================================
# This function predicts diabetes for new patients by:
# 1. Reading the best model path from a text file (or using fallback)
# 2. Loading the trained SVM model
# 3. Creating sample patient data with diabetes features
# 4. Making predictions (handling SVM's lack of probability support)
# 5. Displaying results appropriately
def predict_diabetes():
    """Example: Predict diabetes for new patients."""
    
    # Read the best model path from the text file
    try:
        with open("runs/diabetes/best_model.txt", "r") as f:
            model_path = f.read().strip()
    except FileNotFoundError:
        model_path = "runs/diabetes/svm_best_model.pkl"  # fallback to SVM model
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please run the training pipeline first:")
        print("python disease_prediction_pipeline.py --data diabetes.csv --target Outcome --outdir runs/diabetes")
        return
    
    # Load the trained model
    model = load_model(model_path)
    if model is None:
        return
    
    # Create example new patients with diabetes-related features
    # Features include: pregnancies, glucose, blood pressure, BMI, age, etc.
    # Note: Column names are lowercase to match preprocessing
    new_patients = pd.DataFrame([
        {
            'pregnancies': 1, 'glucose': 85, 'bloodpressure': 66, 'skinthickness': 29,
            'insulin': 0, 'bmi': 26.6, 'diabetespedigreefunction': 0.351, 'age': 31
        },
        {
            'pregnancies': 8, 'glucose': 183, 'bloodpressure': 64, 'skinthickness': 0,
            'insulin': 0, 'bmi': 23.3, 'diabetespedigreefunction': 0.672, 'age': 32
        }
    ])
    
    print("\nDiabetes Prediction")
    print("-" * 40)
    
    try:
        # Make predictions
        predictions = model.predict(new_patients)
        
        # Try to get probabilities, but handle models that don't support it
        # SVM models typically don't provide probability estimates
        try:
            probabilities = model.predict_proba(new_patients)[:, 1]
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                status = "No Diabetes" if pred == 0 else "Diabetes"
                print(f"Patient {i+1}: {status} (Probability: {prob:.2%})")
        except AttributeError:
            # For models without predict_proba (like SVM)
            for i, pred in enumerate(predictions):
                status = "No Diabetes" if pred == 0 else "Diabetes"
                print(f"Patient {i+1}: {status} (No probability available)")
            
    except Exception as e:
        print(f"Error making predictions: {e}")

# =============================================================================
# BREAST CANCER PREDICTION FUNCTION
# =============================================================================
# This function predicts breast cancer for new patients by:
# 1. Reading the best model path from a text file (or using fallback)
# 2. Loading the trained model (Random Forest or Logistic Regression)
# 3. Creating sample patient data with tumor features
# 4. Making predictions and calculating probability scores
# 5. Displaying results (Benign vs Malignant)
def predict_breast_cancer():
    """Example: Predict breast cancer for new patients."""
    
    # Read the best model path from the text file
    try:
        with open("runs/breast_cancer/best_model.txt", "r") as f:
            model_path = f.read().strip()
    except FileNotFoundError:
        model_path = "runs/breast_cancer/rf_best_model.pkl"  # fallback to RF model
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please run the training pipeline first:")
        print("python disease_prediction_pipeline.py --data breast_cancer.csv --target target --outdir runs/breast_cancer")
        return
    
    # Load the trained model
    model = load_model(model_path)
    if model is None:
        return
    
    # Create example new patients with tumor features
    # Features include: radius, texture, perimeter, area, smoothness, etc.
    new_patients = pd.DataFrame([
        {
            'radius_mean': 12.5, 'texture_mean': 17.2, 'perimeter_mean': 78.9, 'area_mean': 450.1,
            'smoothness_mean': 0.09, 'compactness_mean': 0.07, 'concavity_mean': 0.04, 'concave_points_mean': 0.02,
            'symmetry_mean': 0.17, 'fractal_dimension_mean': 0.06
        },
        {
            'radius_mean': 22.5, 'texture_mean': 25.8, 'perimeter_mean': 145.2, 'area_mean': 1200.5,
            'smoothness_mean': 0.15, 'compactness_mean': 0.18, 'concavity_mean': 0.16, 'concave_points_mean': 0.12,
            'symmetry_mean': 0.25, 'fractal_dimension_mean': 0.09
        }
    ])
    
    print("\nBreast Cancer Prediction")
    print("-" * 40)
    
    try:
        # Make predictions and get probability scores
        predictions = model.predict(new_patients)
        probabilities = model.predict_proba(new_patients)[:, 1]
        
        # Display results for each patient
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            status = "Benign" if pred == 0 else "Malignant"
            print(f"Patient {i+1}: {status} (Probability: {prob:.2%})")
            
    except Exception as e:
        print(f"Error making predictions: {e}")

# =============================================================================
# MAIN FUNCTION - ORCHESTRATES ALL PREDICTIONS
# =============================================================================
# This function runs predictions for all three disease types:
# 1. Heart Disease prediction
# 2. Diabetes prediction
# 3. Breast Cancer prediction
# 4. Provides helpful tips for users
def main():
    """Main function to run predictions."""
    
    print("Disease Prediction for New Patients")
    print("=" * 50)
    
    # Check if models exist and run predictions for all three diseases
    predict_heart_disease()
    predict_diabetes()
    predict_breast_cancer()
    
    # Provide helpful tips for users
    print("\n" + "=" * 50)
    print("Tips:")
    print("- Make sure to run the training pipeline first")
    print("- Ensure new patient data has the same features as training data")
    print("- Feature names and data types must match exactly")
    print("- Use the same preprocessing steps for new data")

if __name__ == "__main__":
    main()
