import pickle
import pandas as pd
import numpy as np
import os

def load_disease_models():
    """Load all available disease prediction models"""
    models = {}
    
    # Load combined model info
    try:
        with open('disease_models_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
    except FileNotFoundError:
        print("Error: Model info file not found. Please run train.py first.")
        return {}
    
    # Load individual disease models
    for disease in model_info['diseases']:
        model_filename = f'{disease}_model.pkl'
        try:
            with open(model_filename, 'rb') as f:
                model_data = pickle.load(f)
                models[disease] = model_data
        except FileNotFoundError:
            print(f"Warning: Model file '{model_filename}' not found.")
    
    return models, model_info

def predict_disease(disease_name, features):
    """
    Predict disease for given features
    
    Args:
        disease_name (str): Name of the disease ('heart', 'diabetes', 'breast_cancer')
        features (dict): Dictionary containing feature values
    
    Returns:
        dict: Prediction results
    """
    models, model_info = load_disease_models()
    
    if disease_name not in models:
        print(f"Error: Model for '{disease_name}' not found.")
        return None
    
    model_data = models[disease_name]
    feature_names = model_data['feature_names']
    
    # Create feature vector
    feature_vector = np.array([[features[feature] for feature in feature_names]])
    
    # Scale features if needed
    if model_data['scaler'] is not None:
        feature_vector = model_data['scaler'].transform(feature_vector)
    
    # Make prediction
    model = model_data['model']
    prediction = model.predict(feature_vector)[0]
    probability = model.predict_proba(feature_vector)[0]
    
    return {
        'disease_detected': bool(prediction),
        'probability_disease': probability[1],
        'probability_no_disease': probability[0],
        'model_used': model_data['model_name'],
        'disease_name': disease_name
    }

def interactive_heart_prediction():
    """Interactive heart disease prediction"""
    print("=== Heart Disease Prediction ===")
    print("Enter the following medical information:")
    
    try:
        age = int(input("Age: "))
        sex = int(input("Sex (0=female, 1=male): "))
        cp = int(input("Chest Pain Type (0-3): "))
        trestbps = int(input("Resting Blood Pressure (mm Hg): "))
        chol = int(input("Cholesterol (mg/dl): "))
        fbs = int(input("Fasting Blood Sugar > 120 mg/dl (0=no, 1=yes): "))
        restecg = int(input("Resting ECG Results (0-2): "))
        thalach = int(input("Maximum Heart Rate: "))
        exang = int(input("Exercise Induced Angina (0=no, 1=yes): "))
        oldpeak = float(input("ST Depression: "))
        slope = int(input("Slope of Peak Exercise ST (0-2): "))
        ca = int(input("Number of Major Vessels (0-4): "))
        thal = int(input("Thalassemia (0-3): "))
        
        features = {
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
            'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
            'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
        }
        
        result = predict_disease('heart', features)
        
        if result:
            print("\n=== Prediction Results ===")
            print(f"Heart Disease Detected: {'Yes' if result['disease_detected'] else 'No'}")
            print(f"Probability of Heart Disease: {result['probability_disease']:.2%}")
            print(f"Probability of No Heart Disease: {result['probability_no_disease']:.2%}")
            print(f"Model used: {result['model_used']}")
        else:
            print("Prediction failed.")
            
    except ValueError:
        print("Error: Please enter valid numeric values.")
    except KeyboardInterrupt:
        print("\nPrediction cancelled.")

def interactive_diabetes_prediction():
    """Interactive diabetes prediction"""
    print("=== Diabetes Prediction ===")
    print("Enter the following medical information:")
    
    try:
        pregnancies = int(input("Number of Pregnancies: "))
        glucose = int(input("Glucose Level (mg/dl): "))
        blood_pressure = int(input("Blood Pressure (mm Hg): "))
        skin_thickness = int(input("Skin Thickness (mm): "))
        insulin = int(input("Insulin Level (mu U/ml): "))
        bmi = float(input("BMI: "))
        diabetes_pedigree = float(input("Diabetes Pedigree Function: "))
        age = int(input("Age: "))
        
        features = {
            'pregnancies': pregnancies, 'glucose': glucose, 'blood_pressure': blood_pressure,
            'skin_thickness': skin_thickness, 'insulin': insulin, 'bmi': bmi,
            'diabetes_pedigree': diabetes_pedigree, 'age': age
        }
        
        result = predict_disease('diabetes', features)
        
        if result:
            print("\n=== Prediction Results ===")
            print(f"Diabetes Detected: {'Yes' if result['disease_detected'] else 'No'}")
            print(f"Probability of Diabetes: {result['probability_disease']:.2%}")
            print(f"Probability of No Diabetes: {result['probability_no_disease']:.2%}")
            print(f"Model used: {result['model_used']}")
        else:
            print("Prediction failed.")
            
    except ValueError:
        print("Error: Please enter valid numeric values.")
    except KeyboardInterrupt:
        print("\nPrediction cancelled.")

def interactive_breast_cancer_prediction():
    """Interactive breast cancer prediction"""
    print("=== Breast Cancer Prediction ===")
    print("Enter the following medical information:")
    
    try:
        radius_mean = float(input("Radius Mean: "))
        texture_mean = float(input("Texture Mean: "))
        perimeter_mean = float(input("Perimeter Mean: "))
        area_mean = float(input("Area Mean: "))
        smoothness_mean = float(input("Smoothness Mean: "))
        compactness_mean = float(input("Compactness Mean: "))
        concavity_mean = float(input("Concavity Mean: "))
        concave_points_mean = float(input("Concave Points Mean: "))
        symmetry_mean = float(input("Symmetry Mean: "))
        fractal_dimension_mean = float(input("Fractal Dimension Mean: "))
        
        features = {
            'radius_mean': radius_mean, 'texture_mean': texture_mean,
            'perimeter_mean': perimeter_mean, 'area_mean': area_mean,
            'smoothness_mean': smoothness_mean, 'compactness_mean': compactness_mean,
            'concavity_mean': concavity_mean, 'concave_points_mean': concave_points_mean,
            'symmetry_mean': symmetry_mean, 'fractal_dimension_mean': fractal_dimension_mean
        }
        
        result = predict_disease('breast_cancer', features)
        
        if result:
            print("\n=== Prediction Results ===")
            print(f"Malignant Detected: {'Yes' if result['disease_detected'] else 'No'}")
            print(f"Probability of Malignant: {result['probability_disease']:.2%}")
            print(f"Probability of Benign: {result['probability_no_disease']:.2%}")
            print(f"Model used: {result['model_used']}")
        else:
            print("Prediction failed.")
            
    except ValueError:
        print("Error: Please enter valid numeric values.")
    except KeyboardInterrupt:
        print("\nPrediction cancelled.")

def batch_prediction(data_file, disease_name):
    """
    Make predictions for multiple records from a CSV file
    
    Args:
        data_file (str): Path to CSV file with features
        disease_name (str): Name of the disease to predict
    """
    models, model_info = load_disease_models()
    
    if disease_name not in models:
        print(f"Error: Model for '{disease_name}' not found.")
        return None
    
    try:
        # Load data
        df = pd.read_csv(data_file)
        model_data = models[disease_name]
        required_features = model_data['feature_names']
        
        # Check if all required features are present
        missing_features = set(required_features) - set(df.columns)
        if missing_features:
            print(f"Error: Missing features in CSV: {missing_features}")
            return None
        
        # Prepare features
        X = df[required_features]
        
        # Scale features if needed
        if model_data['scaler'] is not None:
            X = model_data['scaler'].transform(X)
        
        # Make predictions
        model = model_data['model']
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'disease_detected': predictions,
            'probability_disease': probabilities[:, 1],
            'probability_no_disease': probabilities[:, 0]
        })
        
        # Combine with original data
        final_results = pd.concat([df, results], axis=1)
        
        # Save results
        output_file = f'{disease_name}_predictions.csv'
        final_results.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
        
        return final_results
        
    except Exception as e:
        print(f"Error processing batch prediction: {e}")
        return None

def main_menu():
    """Main menu for disease prediction"""
    print("=== Disease Prediction System ===")
    print("Available diseases:")
    print("1. Heart Disease")
    print("2. Diabetes")
    print("3. Breast Cancer")
    print("4. Batch Prediction")
    print("5. Exit")
    
    choice = input("\nChoose option (1-5): ")
    
    if choice == "1":
        interactive_heart_prediction()
    elif choice == "2":
        interactive_diabetes_prediction()
    elif choice == "3":
        interactive_breast_cancer_prediction()
    elif choice == "4":
        print("\n=== Batch Prediction ===")
        print("Available diseases: heart, diabetes, breast_cancer")
        disease = input("Enter disease name: ").strip()
        csv_file = input("Enter path to CSV file: ").strip()
        
        if disease and csv_file:
            batch_prediction(csv_file, disease)
        else:
            print("Invalid input.")
    elif choice == "5":
        print("Goodbye!")
        return False
    else:
        print("Invalid choice. Please select 1-5.")
    
    return True

if __name__ == "__main__":
    running = True
    while running:
        running = main_menu()
        if running:
            input("\nPress Enter to continue...")
