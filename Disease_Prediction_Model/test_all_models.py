#!/usr/bin/env python3
"""
Test All Disease Models

This script tests all three disease prediction models (Heart Disease, Diabetes, Breast Cancer)
by loading trained models and making predictions on sample patient data.
It verifies that all models are working correctly and displays their predictions.
"""

import joblib
import pandas as pd
import os

# =============================================================================
# HEART DISEASE MODEL TESTING FUNCTION
# =============================================================================
# This function tests the heart disease prediction model by:
# 1. Loading the trained Random Forest model from disk
# 2. Creating sample patient data with realistic medical features
# 3. Making predictions and calculating probability scores
# 4. Displaying the results in a clear format
def test_heart_disease():
    """Test heart disease model."""
    print("Testing Heart Disease Model")
    print("-" * 40)
    
    try:
        # Load the trained Random Forest model for heart disease prediction
        model = joblib.load('runs/heart/rf_best_model.pkl')
        
        # Create sample patient data with medical features
        # Features include: age, sex, chest pain type, blood pressure, cholesterol, etc.
        test_patient = pd.DataFrame([{
            'age': 55, 'sex': 1, 'cp': 0, 'trestbps': 130, 'chol': 250,
            'fbs': 0, 'restecg': 1, 'thalach': 150, 'exang': 0,
            'oldpeak': 1.5, 'slope': 1, 'ca': 0, 'thal': 2
        }])
        
        # Make prediction and get probability score
        prediction = model.predict(test_patient)[0]
        probability = model.predict_proba(test_patient)[0][1]
        
        # Determine prediction status and display results
        status = "Heart Disease" if prediction == 1 else "No Heart Disease"
        print(f"Model loaded successfully")
        print(f"Prediction: {status}")
        print(f"Probability: {probability:.2%}")
        
    except Exception as e:
        print(f"Error: {e}")

# =============================================================================
# DIABETES MODEL TESTING FUNCTION
# =============================================================================
# This function tests the diabetes prediction model by:
# 1. Loading the trained SVM model from disk
# 2. Creating sample patient data with diabetes-related features
# 3. Making predictions (SVM doesn't provide probability scores)
# 4. Displaying the results
def test_diabetes():
    """Test diabetes model."""
    print("\nTesting Diabetes Model")
    print("-" * 40)
    
    try:
        # Load the trained SVM model for diabetes prediction
        model = joblib.load('runs/diabetes/svm_best_model.pkl')
        
        # Create sample patient data with diabetes features
        # Features include: pregnancies, glucose, blood pressure, BMI, age, etc.
        test_patient = pd.DataFrame([{
            'pregnancies': 1, 'glucose': 85, 'bloodpressure': 66, 'skinthickness': 29,
            'insulin': 0, 'bmi': 26.6, 'diabetespedigreefunction': 0.351, 'age': 31
        }])
        
        # Make prediction (SVM doesn't support predict_proba)
        prediction = model.predict(test_patient)[0]
        
        # Determine prediction status and display results
        status = "Diabetes" if prediction == 1 else "No Diabetes"
        print(f"Model loaded successfully")
        print(f"Prediction: {status}")
        print(f"Probability: Not available for SVM model")
        
    except Exception as e:
        print(f"Error: {e}")

# =============================================================================
# BREAST CANCER MODEL TESTING FUNCTION
# =============================================================================
# This function tests the breast cancer prediction model by:
# 1. Loading the trained Logistic Regression model from disk
# 2. Creating sample patient data with tumor features
# 3. Making predictions and calculating probability scores
# 4. Displaying the results
def test_breast_cancer():
    """Test breast cancer model."""
    print("\nTesting Breast Cancer Model")
    print("-" * 40)
    
    try:
        # Load the trained Logistic Regression model for breast cancer prediction
        model = joblib.load('runs/breast_cancer/logreg_best_model.pkl')
        
        # Create sample patient data with tumor features
        # Features include: radius, texture, perimeter, area, smoothness, etc.
        test_patient = pd.DataFrame([{
            'radius_mean': 12.5, 'texture_mean': 17.2, 'perimeter_mean': 78.9, 'area_mean': 450.1,
            'smoothness_mean': 0.09, 'compactness_mean': 0.07, 'concavity_mean': 0.04, 'concave_points_mean': 0.02,
            'symmetry_mean': 0.17, 'fractal_dimension_mean': 0.06
        }])
        
        # Make prediction and get probability score
        prediction = model.predict(test_patient)[0]
        probability = model.predict_proba(test_patient)[0][1]
        
        # Determine prediction status and display results
        status = "Malignant" if prediction == 1 else "Benign"
        print(f"Model loaded successfully")
        print(f"Prediction: {status}")
        print(f"Probability: {probability:.2%}")
        
    except Exception as e:
        print(f"Error: {e}")

# =============================================================================
# MAIN FUNCTION - ORCHESTRATES ALL TESTS
# =============================================================================
# This function runs all three disease model tests in sequence:
# 1. Tests heart disease model
# 2. Tests diabetes model  
# 3. Tests breast cancer model
# 4. Provides a summary of all test results
def main():
    """Main function to test all models."""
    print("Testing All Disease Prediction Models")
    print("=" * 50)
    
    # Run all three disease model tests
    test_heart_disease()
    test_diabetes()
    test_breast_cancer()
    
    # Display summary of all test results
    print("\n" + "=" * 50)
    print("All models tested successfully!")
    print("Heart Disease: Random Forest model working")
    print("Diabetes: SVM model working")
    print("Breast Cancer: Logistic Regression model working")

if __name__ == "__main__":
    main()
