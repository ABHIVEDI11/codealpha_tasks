#!/usr/bin/env python3
"""
Create Sample Patient Data for Video Demonstration

This script generates diverse patient data for showcasing the disease prediction model.
"""

import pandas as pd
import numpy as np
import random

def create_heart_disease_patients():
    """Create diverse heart disease patient samples."""
    
    # High-risk patients (likely to have heart disease)
    high_risk_patients = pd.DataFrame([
        {
            'age': 68, 'sex': 1, 'cp': 3, 'trestbps': 180, 'chol': 350,
            'fbs': 1, 'restecg': 2, 'thalach': 110, 'exang': 1,
            'oldpeak': 3.5, 'slope': 0, 'ca': 3, 'thal': 3
        },
        {
            'age': 72, 'sex': 0, 'cp': 2, 'trestbps': 160, 'chol': 320,
            'fbs': 1, 'restecg': 1, 'thalach': 95, 'exang': 1,
            'oldpeak': 4.2, 'slope': 0, 'ca': 2, 'thal': 3
        },
        {
            'age': 55, 'sex': 1, 'cp': 1, 'trestbps': 145, 'chol': 280,
            'fbs': 0, 'restecg': 0, 'thalach': 130, 'exang': 1,
            'oldpeak': 2.8, 'slope': 1, 'ca': 1, 'thal': 2
        }
    ])
    
    # Medium-risk patients
    medium_risk_patients = pd.DataFrame([
        {
            'age': 45, 'sex': 1, 'cp': 0, 'trestbps': 140, 'chol': 250,
            'fbs': 0, 'restecg': 1, 'thalach': 140, 'exang': 0,
            'oldpeak': 1.5, 'slope': 1, 'ca': 0, 'thal': 2
        },
        {
            'age': 58, 'sex': 0, 'cp': 1, 'trestbps': 135, 'chol': 240,
            'fbs': 1, 'restecg': 0, 'thalach': 150, 'exang': 0,
            'oldpeak': 1.2, 'slope': 2, 'ca': 0, 'thal': 2
        }
    ])
    
    # Low-risk patients (unlikely to have heart disease)
    low_risk_patients = pd.DataFrame([
        {
            'age': 35, 'sex': 0, 'cp': 0, 'trestbps': 120, 'chol': 180,
            'fbs': 0, 'restecg': 0, 'thalach': 180, 'exang': 0,
            'oldpeak': 0.0, 'slope': 2, 'ca': 0, 'thal': 2
        },
        {
            'age': 42, 'sex': 1, 'cp': 0, 'trestbps': 110, 'chol': 190,
            'fbs': 0, 'restecg': 1, 'thalach': 170, 'exang': 0,
            'oldpeak': 0.5, 'slope': 2, 'ca': 0, 'thal': 2
        },
        {
            'age': 28, 'sex': 0, 'cp': 0, 'trestbps': 105, 'chol': 160,
            'fbs': 0, 'restecg': 0, 'thalach': 190, 'exang': 0,
            'oldpeak': 0.0, 'slope': 2, 'ca': 0, 'thal': 2
        }
    ])
    
    # Combine all patients
    all_patients = pd.concat([high_risk_patients, medium_risk_patients, low_risk_patients], ignore_index=True)
    
    # Add patient IDs and descriptions
    all_patients['patient_id'] = [f"HD_{i+1:02d}" for i in range(len(all_patients))]
    all_patients['risk_level'] = ['High Risk'] * 3 + ['Medium Risk'] * 2 + ['Low Risk'] * 3
    
    return all_patients

def create_diabetes_patients():
    """Create diverse diabetes patient samples."""
    
    # High-risk patients (likely to have diabetes)
    high_risk_patients = pd.DataFrame([
        {
            'pregnancies': 10, 'glucose': 200, 'bloodpressure': 85, 'skinthickness': 45,
            'insulin': 500, 'bmi': 35.2, 'diabetespedigreefunction': 0.850, 'age': 55
        },
        {
            'pregnancies': 8, 'glucose': 185, 'bloodpressure': 90, 'skinthickness': 40,
            'insulin': 450, 'bmi': 38.1, 'diabetespedigreefunction': 0.920, 'age': 62
        },
        {
            'pregnancies': 12, 'glucose': 220, 'bloodpressure': 88, 'skinthickness': 50,
            'insulin': 600, 'bmi': 40.5, 'diabetespedigreefunction': 1.200, 'age': 58
        }
    ])
    
    # Medium-risk patients
    medium_risk_patients = pd.DataFrame([
        {
            'pregnancies': 5, 'glucose': 140, 'bloodpressure': 75, 'skinthickness': 30,
            'insulin': 200, 'bmi': 28.5, 'diabetespedigreefunction': 0.650, 'age': 45
        },
        {
            'pregnancies': 6, 'glucose': 150, 'bloodpressure': 78, 'skinthickness': 32,
            'insulin': 250, 'bmi': 30.2, 'diabetespedigreefunction': 0.720, 'age': 48
        }
    ])
    
    # Low-risk patients (unlikely to have diabetes)
    low_risk_patients = pd.DataFrame([
        {
            'pregnancies': 1, 'glucose': 85, 'bloodpressure': 66, 'skinthickness': 25,
            'insulin': 80, 'bmi': 24.5, 'diabetespedigreefunction': 0.350, 'age': 30
        },
        {
            'pregnancies': 0, 'glucose': 75, 'bloodpressure': 62, 'skinthickness': 20,
            'insulin': 60, 'bmi': 22.1, 'diabetespedigreefunction': 0.280, 'age': 25
        },
        {
            'pregnancies': 2, 'glucose': 90, 'bloodpressure': 68, 'skinthickness': 28,
            'insulin': 100, 'bmi': 25.8, 'diabetespedigreefunction': 0.420, 'age': 35
        }
    ])
    
    # Combine all patients
    all_patients = pd.concat([high_risk_patients, medium_risk_patients, low_risk_patients], ignore_index=True)
    
    # Add patient IDs and descriptions
    all_patients['patient_id'] = [f"DB_{i+1:02d}" for i in range(len(all_patients))]
    all_patients['risk_level'] = ['High Risk'] * 3 + ['Medium Risk'] * 2 + ['Low Risk'] * 3
    
    return all_patients

def create_breast_cancer_patients():
    """Create diverse breast cancer patient samples."""
    
    # High-risk patients (likely to have malignant tumors)
    high_risk_patients = pd.DataFrame([
        {
            'radius_mean': 22.5, 'texture_mean': 25.8, 'perimeter_mean': 145.2, 'area_mean': 1200.5,
            'smoothness_mean': 0.15, 'compactness_mean': 0.18, 'concavity_mean': 0.16, 'concave_points_mean': 0.12,
            'symmetry_mean': 0.25, 'fractal_dimension_mean': 0.09
        },
        {
            'radius_mean': 20.8, 'texture_mean': 24.1, 'perimeter_mean': 135.7, 'area_mean': 1100.2,
            'smoothness_mean': 0.14, 'compactness_mean': 0.17, 'concavity_mean': 0.15, 'concave_points_mean': 0.11,
            'symmetry_mean': 0.24, 'fractal_dimension_mean': 0.08
        },
        {
            'radius_mean': 24.2, 'texture_mean': 27.3, 'perimeter_mean': 155.8, 'area_mean': 1350.1,
            'smoothness_mean': 0.16, 'compactness_mean': 0.19, 'concavity_mean': 0.17, 'concave_points_mean': 0.13,
            'symmetry_mean': 0.26, 'fractal_dimension_mean': 0.10
        }
    ])
    
    # Medium-risk patients
    medium_risk_patients = pd.DataFrame([
        {
            'radius_mean': 16.5, 'texture_mean': 20.2, 'perimeter_mean': 105.3, 'area_mean': 750.8,
            'smoothness_mean': 0.11, 'compactness_mean': 0.12, 'concavity_mean': 0.08, 'concave_points_mean': 0.05,
            'symmetry_mean': 0.20, 'fractal_dimension_mean': 0.07
        },
        {
            'radius_mean': 15.8, 'texture_mean': 19.5, 'perimeter_mean': 98.7, 'area_mean': 680.4,
            'smoothness_mean': 0.10, 'compactness_mean': 0.11, 'concavity_mean': 0.07, 'concave_points_mean': 0.04,
            'symmetry_mean': 0.19, 'fractal_dimension_mean': 0.06
        }
    ])
    
    # Low-risk patients (likely to have benign tumors)
    low_risk_patients = pd.DataFrame([
        {
            'radius_mean': 11.2, 'texture_mean': 16.8, 'perimeter_mean': 72.5, 'area_mean': 420.3,
            'smoothness_mean': 0.08, 'compactness_mean': 0.06, 'concavity_mean': 0.03, 'concave_points_mean': 0.02,
            'symmetry_mean': 0.16, 'fractal_dimension_mean': 0.05
        },
        {
            'radius_mean': 10.8, 'texture_mean': 15.9, 'perimeter_mean': 68.2, 'area_mean': 380.7,
            'smoothness_mean': 0.07, 'compactness_mean': 0.05, 'concavity_mean': 0.02, 'concave_points_mean': 0.01,
            'symmetry_mean': 0.15, 'fractal_dimension_mean': 0.04
        },
        {
            'radius_mean': 12.5, 'texture_mean': 17.2, 'perimeter_mean': 78.9, 'area_mean': 450.1,
            'smoothness_mean': 0.09, 'compactness_mean': 0.07, 'concavity_mean': 0.04, 'concave_points_mean': 0.02,
            'symmetry_mean': 0.17, 'fractal_dimension_mean': 0.06
        }
    ])
    
    # Combine all patients
    all_patients = pd.concat([high_risk_patients, medium_risk_patients, low_risk_patients], ignore_index=True)
    
    # Add patient IDs and descriptions
    all_patients['patient_id'] = [f"BC_{i+1:02d}" for i in range(len(all_patients))]
    all_patients['risk_level'] = ['High Risk'] * 3 + ['Medium Risk'] * 2 + ['Low Risk'] * 3
    
    return all_patients

def main():
    """Create sample patient data for video demonstration."""
    
    print("🏥 Creating Sample Patient Data for Video Demonstration")
    print("=" * 60)
    
    # Create heart disease patients
    heart_patients = create_heart_disease_patients()
    heart_patients.to_csv("sample_heart_patients.csv", index=False)
    print(f"✅ Created {len(heart_patients)} heart disease patient samples")
    print("   - 3 High Risk patients")
    print("   - 2 Medium Risk patients") 
    print("   - 3 Low Risk patients")
    
    # Create diabetes patients
    diabetes_patients = create_diabetes_patients()
    diabetes_patients.to_csv("sample_diabetes_patients.csv", index=False)
    print(f"✅ Created {len(diabetes_patients)} diabetes patient samples")
    print("   - 3 High Risk patients")
    print("   - 2 Medium Risk patients")
    print("   - 3 Low Risk patients")
    
    # Create breast cancer patients
    breast_cancer_patients = create_breast_cancer_patients()
    breast_cancer_patients.to_csv("sample_breast_cancer_patients.csv", index=False)
    print(f"✅ Created {len(breast_cancer_patients)} breast cancer patient samples")
    print("   - 3 High Risk patients")
    print("   - 2 Medium Risk patients")
    print("   - 3 Low Risk patients")
    
    print("\n📊 Sample Patient Overview:")
    print("-" * 40)
    
    print("\nHeart Disease Patients:")
    for _, patient in heart_patients.iterrows():
        print(f"  {patient['patient_id']}: {patient['risk_level']} - Age: {patient['age']}, BP: {patient['trestbps']}, Cholesterol: {patient['chol']}")
    
    print("\nDiabetes Patients:")
    for _, patient in diabetes_patients.iterrows():
        print(f"  {patient['patient_id']}: {patient['risk_level']} - Age: {patient['age']}, Glucose: {patient['glucose']}, BMI: {patient['bmi']}")
    
    print("\nBreast Cancer Patients:")
    for _, patient in breast_cancer_patients.iterrows():
        print(f"  {patient['patient_id']}: {patient['risk_level']} - Radius: {patient['radius_mean']:.1f}, Area: {patient['area_mean']:.0f}, Texture: {patient['texture_mean']:.1f}")
    
    print("\n🎥 Video Demonstration Ready!")
    print("Files created:")
    print("- sample_heart_patients.csv")
    print("- sample_diabetes_patients.csv")
    print("- sample_breast_cancer_patients.csv")

if __name__ == "__main__":
    main()
