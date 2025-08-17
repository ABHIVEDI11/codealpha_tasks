import os
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
import numpy as np

def create_heart_disease_data():
    """Create Heart Disease dataset using sklearn-like data"""
    print("Creating Heart Disease dataset...")
    
    try:
        np.random.seed(42)
        n_samples = 303  # Same as original Cleveland dataset
        
        # Generate realistic heart disease features
        age = np.random.normal(54, 9, n_samples).astype(int)
        age = np.clip(age, 29, 77)
        
        sex = np.random.choice([0, 1], n_samples, p=[0.68, 0.32])  # 68% male, 32% female
        
        cp = np.random.choice([0, 1, 2, 3], n_samples, p=[0.46, 0.16, 0.28, 0.10])  # chest pain type
        
        trestbps = np.random.normal(131, 18, n_samples).astype(int)  # resting blood pressure
        trestbps = np.clip(trestbps, 94, 200)
        
        chol = np.random.normal(246, 51, n_samples).astype(int)  # cholesterol
        chol = np.clip(chol, 126, 564)
        
        fbs = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])  # fasting blood sugar
        
        restecg = np.random.choice([0, 1, 2], n_samples, p=[0.48, 0.42, 0.10])  # resting ecg
        
        thalach = np.random.normal(149, 23, n_samples).astype(int)  # max heart rate
        thalach = np.clip(thalach, 71, 202)
        
        exang = np.random.choice([0, 1], n_samples, p=[0.67, 0.33])  # exercise induced angina
        
        oldpeak = np.random.exponential(1.04, n_samples)  # ST depression
        oldpeak = np.clip(oldpeak, 0, 6.2)
        
        slope = np.random.choice([0, 1, 2], n_samples, p=[0.48, 0.42, 0.10])  # slope
        
        ca = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.58, 0.19, 0.12, 0.08, 0.03])  # vessels
        
        thal = np.random.choice([0, 1, 2, 3], n_samples, p=[0.48, 0.42, 0.08, 0.02])  # thalassemia
        
        # Create target based on risk factors
        risk_score = (age * 0.1 + sex * 0.2 + cp * 0.3 + (trestbps - 120) * 0.01 + 
                     (chol - 200) * 0.001 + fbs * 0.2 + restecg * 0.1 + 
                     (thalach < 150) * 0.3 + exang * 0.4 + oldpeak * 0.2 + 
                     slope * 0.1 + ca * 0.3 + thal * 0.2)
        
        target = (risk_score > np.median(risk_score)).astype(int)
        
        # Create DataFrame
        data = pd.DataFrame({
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
            'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
            'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal,
            'target': target
        })
        
        # Save to CSV
        data.to_csv('Disease_Prediction_Model/heart_disease.csv', index=False)
        
        print("Heart Disease dataset created successfully!")
        return True
    except Exception as e:
        print(f"Error creating Heart Disease dataset: {e}")
        return False

def create_diabetes_data():
    """Create Diabetes dataset using sklearn-like data"""
    print("Creating Diabetes dataset...")
    
    try:
        np.random.seed(42)
        n_samples = 768  # Same as original diabetes dataset
        
        # Generate realistic diabetes features
        pregnancies = np.random.poisson(3.8, n_samples)
        pregnancies = np.clip(pregnancies, 0, 17)
        
        glucose = np.random.normal(120, 32, n_samples).astype(int)
        glucose = np.clip(glucose, 44, 199)
        
        blood_pressure = np.random.normal(69, 19, n_samples).astype(int)
        blood_pressure = np.clip(blood_pressure, 24, 122)
        
        skin_thickness = np.random.exponential(20, n_samples).astype(int)
        skin_thickness = np.clip(skin_thickness, 7, 99)
        
        insulin = np.random.exponential(79, n_samples).astype(int)
        insulin = np.clip(insulin, 14, 846)
        
        bmi = np.random.normal(32, 7.9, n_samples)
        bmi = np.clip(bmi, 18.2, 67.1)
        
        diabetes_pedigree = np.random.exponential(0.5, n_samples)
        diabetes_pedigree = np.clip(diabetes_pedigree, 0.078, 2.42)
        
        age = np.random.normal(33, 11.8, n_samples).astype(int)
        age = np.clip(age, 21, 81)
        
        # Create target based on risk factors
        risk_score = (pregnancies * 0.1 + (glucose - 100) * 0.01 + 
                     (blood_pressure - 80) * 0.01 + (skin_thickness - 30) * 0.01 +
                     (insulin - 100) * 0.001 + (bmi - 25) * 0.05 + 
                     diabetes_pedigree * 0.5 + age * 0.01)
        
        target = (risk_score > np.median(risk_score)).astype(int)
        
        # Create DataFrame
        data = pd.DataFrame({
            'pregnancies': pregnancies, 'glucose': glucose, 'blood_pressure': blood_pressure,
            'skin_thickness': skin_thickness, 'insulin': insulin, 'bmi': bmi,
            'diabetes_pedigree': diabetes_pedigree, 'age': age, 'target': target
        })
        
        # Save to CSV
        data.to_csv('Disease_Prediction_Model/diabetes.csv', index=False)
        
        print("Diabetes dataset created successfully!")
        return True
    except Exception as e:
        print(f"Error creating Diabetes dataset: {e}")
        return False

def create_breast_cancer_data():
    """Create Breast Cancer dataset using sklearn"""
    print("Creating Breast Cancer dataset...")
    
    try:
        # Load from sklearn
        cancer = load_breast_cancer()
        
        # Create DataFrame
        df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
        df['target'] = cancer.target
        
        # Save to CSV
        df.to_csv('Disease_Prediction_Model/breast_cancer.csv', index=False)
        
        print("Breast Cancer dataset created successfully!")
        return True
    except Exception as e:
        print(f"Error creating Breast Cancer dataset: {e}")
        return False

def create_credit_data():
    """Create realistic credit scoring dataset"""
    print("Creating Credit Scoring dataset...")
    
    try:
        # Generate realistic credit data
        np.random.seed(42)
        n_samples = 1000
        
        # Generate more realistic features with correlations
        age = np.random.normal(35, 12, n_samples)
        age = np.clip(age, 18, 75).astype(int)
        
        # Income correlated with age
        base_income = 30000 + age * 800
        income = np.random.normal(base_income, 15000, n_samples)
        income = np.clip(income, 15000, 200000)
        
        # Employment length correlated with age
        max_employment = np.maximum(0, age - 18)
        employment_length = np.random.uniform(0, max_employment, n_samples)
        employment_length = np.clip(employment_length, 0, 40).astype(int)
        
        # Debt correlated with income
        debt_ratio = np.random.normal(0.3, 0.2, n_samples)
        debt_ratio = np.clip(debt_ratio, 0, 1)
        debt = income * debt_ratio
        debt = np.clip(debt, 0, 100000)
        
        # Payment history correlated with income and age
        base_payment = 70 + (income - 50000) / 1000 + (age - 35) / 2
        payment_history = np.random.normal(base_payment, 15, n_samples)
        payment_history = np.clip(payment_history, 0, 100).astype(int)
        
        # Credit utilization correlated with debt and income
        credit_limit = income * 0.3
        credit_utilization = debt / (credit_limit + 1)
        credit_utilization = np.clip(credit_utilization, 0, 1)
        
        # Credit inquiries
        credit_inquiries = np.random.poisson(2, n_samples)
        credit_inquiries = np.clip(credit_inquiries, 0, 10)
        
        # Create target variable using realistic credit scoring logic
        payment_factor = payment_history * 0.35
        utilization_factor = (1 - credit_utilization) * 100 * 0.30
        length_factor = np.minimum(employment_length * 10, 100) * 0.15
        income_factor = np.minimum((income - 30000) / 1000, 50) * 0.10
        inquiry_factor = (10 - credit_inquiries) * 10 * 0.10
        
        credit_score = payment_factor + utilization_factor + length_factor + income_factor + inquiry_factor
        target = (credit_score > 60).astype(int)
        
        # Create DataFrame
        data = pd.DataFrame({
            'income': income,
            'debt': debt,
            'payment_history': payment_history,
            'credit_utilization': credit_utilization,
            'age': age,
            'employment_length': employment_length,
            'credit_inquiries': credit_inquiries,
            'creditworthy': target
        })
        
        # Save to CSV
        data.to_csv('Credit_Scoring_Model/credit_data.csv', index=False)
        
        print("Credit Scoring dataset created successfully!")
        return True
    except Exception as e:
        print(f"Error creating Credit Scoring dataset: {e}")
        return False

def create_audio_samples():
    """Create sample audio files for emotion recognition (simplified)"""
    print("Creating sample audio files for emotion recognition...")
    
    try:
        # Create results directory
        os.makedirs('Emotion_Prediction_Moedel/audio_samples', exist_ok=True)
        
        emotions = ['happy', 'sad', 'angry', 'neutral', 'fearful', 'surprised']
        sample_rate = 22050
        duration = 1  # Reduced to 1 second for faster generation
        
        for emotion in emotions:
            print(f"Creating {emotion} audio sample...")
            
            # Generate simple audio signal
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            if emotion == 'happy':
                audio = 0.3 * np.sin(2 * np.pi * 400 * t) + 0.2 * np.sin(2 * np.pi * 600 * t)
            elif emotion == 'sad':
                audio = 0.2 * np.sin(2 * np.pi * 200 * t) + 0.1 * np.sin(2 * np.pi * 300 * t)
            elif emotion == 'angry':
                audio = 0.4 * np.sin(2 * np.pi * 500 * t) + 0.3 * np.sin(2 * np.pi * 800 * t)
            elif emotion == 'fearful':
                audio = 0.25 * np.sin(2 * np.pi * 250 * t) + 0.15 * np.sin(2 * np.pi * 350 * t)
            elif emotion == 'surprised':
                audio = 0.35 * np.sin(2 * np.pi * 450 * t) + 0.25 * np.sin(2 * np.pi * 700 * t)
            else:  # neutral
                audio = 0.3 * np.sin(2 * np.pi * 300 * t) + 0.2 * np.sin(2 * np.pi * 500 * t)
            
            # Add some noise
            audio += np.random.normal(0, 0.01, len(audio))
            
            # Normalize
            audio = audio / np.max(np.abs(audio))
            
            # Save as numpy array (faster than WAV)
            filename = f'Emotion_Prediction_Moedel/audio_samples/{emotion}_sample.npy'
            np.save(filename, audio)
        
        print("Audio samples created successfully!")
        return True
    except Exception as e:
        print(f"Error creating audio samples: {e}")
        return False

def main():
    """Create all datasets quickly"""
    print("Starting dataset creation...")
    
    # Create directories if they don't exist
    os.makedirs('Disease_Prediction_Model', exist_ok=True)
    os.makedirs('Credit_Scoring_Model', exist_ok=True)
    os.makedirs('Emotion_Prediction_Moedel', exist_ok=True)
    
    # Create datasets
    success_count = 0
    
    if create_heart_disease_data():
        success_count += 1
    
    if create_diabetes_data():
        success_count += 1
    
    if create_breast_cancer_data():
        success_count += 1
    
    if create_credit_data():
        success_count += 1
    
    if create_audio_samples():
        success_count += 1
    
    print(f"\nDataset creation completed! {success_count}/5 datasets successfully created.")
    print("\nDatasets available:")
    print("- Disease_Prediction_Model/heart_disease.csv")
    print("- Disease_Prediction_Model/diabetes.csv") 
    print("- Disease_Prediction_Model/breast_cancer.csv")
    print("- Credit_Scoring_Model/credit_data.csv")
    print("- Emotion_Prediction_Moedel/audio_samples/ (audio files)")

if __name__ == "__main__":
    main()
