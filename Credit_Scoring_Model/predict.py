import pickle
import pandas as pd
import numpy as np

def load_model():
    """Load the trained credit scoring model"""
    try:
        with open('credit_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        print("Error: Model file 'credit_model.pkl' not found. Please run train.py first.")
        return None

def predict_creditworthiness(features):
    """
    Predict creditworthiness for given features
    
    Args:
        features (dict): Dictionary containing:
            - income: Annual income
            - debt: Total debt
            - payment_history: Payment history score (0-100)
            - credit_utilization: Credit utilization ratio (0-1)
            - age: Age of applicant
            - employment_length: Years of employment
    
    Returns:
        dict: Prediction results
    """
    model_data = load_model()
    if model_data is None:
        return None
    
    # Create feature vector
    feature_names = model_data['feature_names']
    feature_vector = np.array([[features[feature] for feature in feature_names]])
    
    # Scale features if needed
    if model_data['scaler'] is not None:
        feature_vector = model_data['scaler'].transform(feature_vector)
    
    # Make prediction
    model = model_data['model']
    prediction = model.predict(feature_vector)[0]
    probability = model.predict_proba(feature_vector)[0]
    
    return {
        'creditworthy': bool(prediction),
        'probability_creditworthy': probability[1],
        'probability_not_creditworthy': probability[0],
        'model_used': model_data['model_name']
    }

def interactive_prediction():
    """Interactive prediction interface"""
    print("=== Credit Scoring Prediction ===")
    print("Enter the following information:")
    
    try:
        income = float(input("Annual Income ($): "))
        debt = float(input("Total Debt ($): "))
        payment_history = float(input("Payment History Score (0-100): "))
        credit_utilization = float(input("Credit Utilization Ratio (0-1): "))
        age = int(input("Age: "))
        employment_length = int(input("Years of Employment: "))
        credit_inquiries = int(input("Number of Credit Inquiries (0-10): "))
        
        features = {
            'income': income,
            'debt': debt,
            'payment_history': payment_history,
            'credit_utilization': credit_utilization,
            'age': age,
            'employment_length': employment_length,
            'credit_inquiries': credit_inquiries
        }
        
        result = predict_creditworthiness(features)
        
        if result:
            print("\n=== Prediction Results ===")
            print(f"Creditworthy: {'Yes' if result['creditworthy'] else 'No'}")
            print(f"Probability of being creditworthy: {result['probability_creditworthy']:.2%}")
            print(f"Probability of not being creditworthy: {result['probability_not_creditworthy']:.2%}")
            print(f"Model used: {result['model_used']}")
        else:
            print("Prediction failed. Please check if the model is trained.")
            
    except ValueError:
        print("Error: Please enter valid numeric values.")
    except KeyboardInterrupt:
        print("\nPrediction cancelled.")

def batch_prediction(data_file):
    """
    Make predictions for multiple records from a CSV file
    
    Args:
        data_file (str): Path to CSV file with features
    """
    model_data = load_model()
    if model_data is None:
        return None
    
    try:
        # Load data
        df = pd.read_csv(data_file)
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
            'creditworthy': predictions,
            'probability_creditworthy': probabilities[:, 1],
            'probability_not_creditworthy': probabilities[:, 0]
        })
        
        # Combine with original data
        final_results = pd.concat([df, results], axis=1)
        
        # Save results
        output_file = 'credit_predictions.csv'
        final_results.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
        
        return final_results
        
    except Exception as e:
        print(f"Error processing batch prediction: {e}")
        return None

if __name__ == "__main__":
    print("Credit Scoring Prediction System")
    print("1. Interactive prediction")
    print("2. Batch prediction from CSV file")
    
    choice = input("Choose option (1 or 2): ")
    
    if choice == "1":
        interactive_prediction()
    elif choice == "2":
        csv_file = input("Enter path to CSV file: ")
        batch_prediction(csv_file)
    else:
        print("Invalid choice. Please run again and select 1 or 2.")
