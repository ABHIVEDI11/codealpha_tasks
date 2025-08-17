#!/usr/bin/env python3
"""
Video Demonstration - Disease Prediction with Multiple Patients

This script demonstrates the disease prediction model with diverse patient scenarios
for video creation.
"""

import joblib
import pandas as pd
import os
import sys

def load_model(model_path):
    """Load a trained model from pickle file."""
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def predict_heart_disease_batch():
    """Predict heart disease for multiple patients."""
    
    # Load the best heart disease model
    try:
        with open("runs/heart/best_model.txt", "r") as f:
            model_path = f.read().strip()
    except FileNotFoundError:
        model_path = "runs/heart/rf_best_model.pkl"
    
    if not os.path.exists(model_path):
        print("❌ Heart disease model not found. Please train the model first.")
        return
    
    model = load_model(model_path)
    if model is None:
        return
    
    # Load sample patients
    try:
        patients = pd.read_csv("sample_heart_patients.csv")
    except FileNotFoundError:
        print("❌ Sample heart patients file not found. Run create_sample_patients.py first.")
        return
    
    print("❤️ HEART DISEASE PREDICTION DEMO")
    print("=" * 50)
    
    # Remove non-feature columns for prediction
    feature_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                      'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    prediction_data = patients[feature_columns]
    
    try:
        predictions = model.predict(prediction_data)
        
        # Try to get probabilities
        try:
            probabilities = model.predict_proba(prediction_data)[:, 1]
            has_probabilities = True
        except AttributeError:
            has_probabilities = False
        
        print(f"{'Patient ID':<10} {'Risk Level':<12} {'Prediction':<15} {'Probability':<12} {'Details'}")
        print("-" * 80)
        
        for i, (_, patient) in enumerate(patients.iterrows()):
            pred = predictions[i]
            status = "🔴 HEART DISEASE" if pred == 1 else "🟢 NO HEART DISEASE"
            
            if has_probabilities:
                prob = probabilities[i]
                prob_str = f"{prob:.1%}"
            else:
                prob_str = "N/A"
            
            details = f"Age: {patient['age']}, BP: {patient['trestbps']}, Chol: {patient['chol']}"
            
            print(f"{patient['patient_id']:<10} {patient['risk_level']:<12} {status:<15} {prob_str:<12} {details}")
            
    except Exception as e:
        print(f"❌ Error making predictions: {e}")

def predict_diabetes_batch():
    """Predict diabetes for multiple patients."""
    
    # Load the best diabetes model
    try:
        with open("runs/diabetes/best_model.txt", "r") as f:
            model_path = f.read().strip()
    except FileNotFoundError:
        model_path = "runs/diabetes/svm_best_model.pkl"
    
    if not os.path.exists(model_path):
        print("❌ Diabetes model not found. Please train the model first.")
        return
    
    model = load_model(model_path)
    if model is None:
        return
    
    # Load sample patients
    try:
        patients = pd.read_csv("sample_diabetes_patients.csv")
    except FileNotFoundError:
        print("❌ Sample diabetes patients file not found. Run create_sample_patients.py first.")
        return
    
    print("\n🩸 DIABETES PREDICTION DEMO")
    print("=" * 50)
    
    # Remove non-feature columns for prediction
    feature_columns = ['pregnancies', 'glucose', 'bloodpressure', 'skinthickness', 
                      'insulin', 'bmi', 'diabetespedigreefunction', 'age']
    prediction_data = patients[feature_columns]
    
    try:
        predictions = model.predict(prediction_data)
        
        # Try to get probabilities
        try:
            probabilities = model.predict_proba(prediction_data)[:, 1]
            has_probabilities = True
        except AttributeError:
            has_probabilities = False
        
        print(f"{'Patient ID':<10} {'Risk Level':<12} {'Prediction':<15} {'Probability':<12} {'Details'}")
        print("-" * 80)
        
        for i, (_, patient) in enumerate(patients.iterrows()):
            pred = predictions[i]
            status = "🔴 DIABETES" if pred == 1 else "🟢 NO DIABETES"
            
            if has_probabilities:
                prob = probabilities[i]
                prob_str = f"{prob:.1%}"
            else:
                prob_str = "N/A"
            
            details = f"Age: {patient['age']}, Glucose: {patient['glucose']}, BMI: {patient['bmi']}"
            
            print(f"{patient['patient_id']:<10} {patient['risk_level']:<12} {status:<15} {prob_str:<12} {details}")
            
    except Exception as e:
        print(f"❌ Error making predictions: {e}")

def predict_breast_cancer_batch():
    """Predict breast cancer for multiple patients."""
    
    # Load the best breast cancer model
    try:
        with open("runs/breast_cancer/best_model.txt", "r") as f:
            model_path = f.read().strip()
    except FileNotFoundError:
        model_path = "runs/breast_cancer/rf_best_model.pkl"
    
    if not os.path.exists(model_path):
        print("❌ Breast cancer model not found. Please train the model first.")
        return
    
    model = load_model(model_path)
    if model is None:
        return
    
    # Load sample patients
    try:
        patients = pd.read_csv("sample_breast_cancer_patients.csv")
    except FileNotFoundError:
        print("❌ Sample breast cancer patients file not found. Run create_sample_patients.py first.")
        return
    
    print("\n🩺 BREAST CANCER PREDICTION DEMO")
    print("=" * 50)
    
    # Remove non-feature columns for prediction
    feature_columns = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
                      'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean',
                      'symmetry_mean', 'fractal_dimension_mean']
    prediction_data = patients[feature_columns]
    
    try:
        predictions = model.predict(prediction_data)
        
        # Try to get probabilities
        try:
            probabilities = model.predict_proba(prediction_data)[:, 1]
            has_probabilities = True
        except AttributeError:
            has_probabilities = False
        
        print(f"{'Patient ID':<10} {'Risk Level':<12} {'Prediction':<15} {'Probability':<12} {'Details'}")
        print("-" * 80)
        
        for i, (_, patient) in enumerate(patients.iterrows()):
            pred = predictions[i]
            status = "🔴 MALIGNANT" if pred == 1 else "🟢 BENIGN"
            
            if has_probabilities:
                prob = probabilities[i]
                prob_str = f"{prob:.1%}"
            else:
                prob_str = "N/A"
            
            details = f"Radius: {patient['radius_mean']:.1f}, Area: {patient['area_mean']:.0f}, Texture: {patient['texture_mean']:.1f}"
            
            print(f"{patient['patient_id']:<10} {patient['risk_level']:<12} {status:<15} {prob_str:<12} {details}")
            
    except Exception as e:
        print(f"❌ Error making predictions: {e}")

def show_model_performance():
    """Show model performance metrics."""
    
    print("\n📊 MODEL PERFORMANCE SUMMARY")
    print("=" * 50)
    
    # Heart Disease Performance
    try:
        heart_metrics = pd.read_csv("runs/heart/metrics.csv")
        print("\n❤️ Heart Disease Model Performance:")
        print("-" * 40)
        for _, row in heart_metrics.iterrows():
            print(f"{row['model'].upper():<8} - Accuracy: {row['accuracy']:.1%}, ROC AUC: {row['roc_auc']:.1%}")
    except FileNotFoundError:
        print("❌ Heart disease metrics not found")
    
    # Diabetes Performance
    try:
        diabetes_metrics = pd.read_csv("runs/diabetes/metrics.csv")
        print("\n🩸 Diabetes Model Performance:")
        print("-" * 40)
        for _, row in diabetes_metrics.iterrows():
            print(f"{row['model'].upper():<8} - Accuracy: {row['accuracy']:.1%}, ROC AUC: {row['roc_auc']:.1%}")
    except FileNotFoundError:
        print("❌ Diabetes metrics not found")
    
    # Breast Cancer Performance
    try:
        breast_cancer_metrics = pd.read_csv("runs/breast_cancer/metrics.csv")
        print("\n🩺 Breast Cancer Model Performance:")
        print("-" * 40)
        for _, row in breast_cancer_metrics.iterrows():
            print(f"{row['model'].upper():<8} - Accuracy: {row['accuracy']:.1%}, ROC AUC: {row['roc_auc']:.1%}")
    except FileNotFoundError:
        print("❌ Breast cancer metrics not found")

def main():
    """Main function for video demonstration."""
    
    print("🎥 DISEASE PREDICTION MODEL - VIDEO DEMONSTRATION")
    print("=" * 60)
    print("This demo shows predictions for multiple patients with different risk levels")
    print("Perfect for showcasing the model's capabilities in a video presentation")
    print("=" * 60)
    
    # Show model performance first
    show_model_performance()
    
    # Run predictions
    predict_heart_disease_batch()
    predict_diabetes_batch()
    predict_breast_cancer_batch()
    
    print("\n" + "=" * 60)
    print("🎬 VIDEO DEMONSTRATION COMPLETE!")
    print("=" * 60)
    print("Key Points for Your Video:")
    print("✅ Model successfully predicts disease risk")
    print("✅ Handles different patient risk levels")
    print("✅ Provides probability scores (when available)")
    print("✅ Shows real-world medical applications")
    print("✅ Clean, professional output format")
    print("✅ Covers three major disease types")

if __name__ == "__main__":
    main()
