import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

def load_datasets():
    """Load and prepare disease datasets"""
    datasets = {}
    
    # Heart Disease Dataset
    try:
        if os.path.exists('heart_disease.csv'):
            # Load heart disease dataset
            heart_data = pd.read_csv('heart_disease.csv')
            
            # Clean data - remove rows with missing values
            heart_data = heart_data.dropna()
            
            datasets['heart'] = heart_data
            print(f"Heart disease dataset loaded: {len(heart_data)} samples")
            
    except Exception as e:
        print(f"Error loading heart dataset: {e}")
    
    # Diabetes Dataset
    try:
        if os.path.exists('diabetes.csv'):
            # Load diabetes dataset
            diabetes_data = pd.read_csv('diabetes.csv')
            
            # Clean data - remove rows with missing values
            diabetes_data = diabetes_data.dropna()
            
            datasets['diabetes'] = diabetes_data
            print(f"Diabetes dataset loaded: {len(diabetes_data)} samples")
            
    except Exception as e:
        print(f"Error loading diabetes dataset: {e}")
    
    # Breast Cancer Dataset
    try:
        if os.path.exists('breast_cancer.csv'):
            # Load breast cancer dataset
            breast_data = pd.read_csv('breast_cancer.csv')
            
            # Clean data - remove rows with missing values
            breast_data = breast_data.dropna()
            
            datasets['breast_cancer'] = breast_data
            print(f"Breast cancer dataset loaded: {len(breast_data)} samples")
            
    except Exception as e:
        print(f"Error loading breast cancer dataset: {e}")
    
    return datasets

def train_disease_models():
    """Train models for disease prediction"""
    print("Loading datasets...")
    datasets = load_datasets()
    
    if not datasets:
        print("No datasets available. Exiting.")
        return
    
    all_results = {}
    
    for disease_name, data in datasets.items():
        print(f"\n{'='*50}")
        print(f"Training models for {disease_name.upper()} prediction")
        print(f"{'='*50}")
        
        # Prepare data
        X = data.drop('target', axis=1)
        y = data['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            if name == 'SVM':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            elif name == 'Logistic Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:  # Random Forest
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'model': model,
                'scaler': scaler if name in ['Logistic Regression', 'SVM'] else None,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'roc_auc': roc_auc
            }
            
            print(f"{name} ROC-AUC: {roc_auc:.4f}")
            print(f"{name} Classification Report:")
            print(classification_report(y_test, y_pred))
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['roc_auc'])
        best_model = results[best_model_name]['model']
        best_scaler = results[best_model_name]['scaler']
        
        print(f"\nBest model for {disease_name}: {best_model_name} (ROC-AUC: {results[best_model_name]['roc_auc']:.4f})")
        
        # Save best model
        model_data = {
            'model': best_model,
            'scaler': best_scaler,
            'feature_names': X.columns.tolist(),
            'model_name': best_model_name,
            'disease_name': disease_name
        }
        
        model_filename = f'{disease_name}_model.pkl'
        with open(model_filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved as '{model_filename}'")
        
        # Create visualizations
        create_disease_visualizations(results, y_test, X.columns, disease_name)
        
        all_results[disease_name] = results
    
    # Save combined model info
    combined_info = {
        'diseases': list(datasets.keys()),
        'feature_counts': {name: len(data.columns) - 1 for name, data in datasets.items()},
        'sample_counts': {name: len(data) for name, data in datasets.items()}
    }
    
    with open('disease_models_info.pkl', 'wb') as f:
        pickle.dump(combined_info, f)
    
    print(f"\nCombined model info saved as 'disease_models_info.pkl'")
    
    return all_results

def create_disease_visualizations(results, y_test, feature_names, disease_name):
    """Create visualizations for disease prediction results"""
    plt.figure(figsize=(15, 5))
    
    # ROC Curves
    plt.subplot(1, 3, 1)
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
        plt.plot(fpr, tpr, label=f'{name} (AUC = {result["roc_auc"]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {disease_name.title()}')
    plt.legend()
    plt.grid(True)
    
    # Feature Importance (for Random Forest)
    if 'Random Forest' in results:
        rf_model = results['Random Forest']['model']
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.subplot(1, 3, 2)
        plt.title(f'Feature Importance - {disease_name.title()}')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.ylabel('Importance')
    
    # Confusion Matrix for best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['roc_auc'])
    y_pred = results[best_model_name]['y_pred']
    
    plt.subplot(1, 3, 3)
    cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {best_model_name}')
    
    plt.tight_layout()
    plt.savefig(f'runs/{disease_name}/disease_model_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Create runs directory structure
    os.makedirs('runs', exist_ok=True)
    for disease in ['heart', 'diabetes', 'breast_cancer']:
        os.makedirs(f'runs/{disease}', exist_ok=True)
    
    train_disease_models()
