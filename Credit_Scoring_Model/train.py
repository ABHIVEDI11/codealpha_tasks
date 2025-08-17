import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def generate_credit_data(n_samples=1000):
    """Generate realistic synthetic credit scoring data"""
    np.random.seed(42)
    
    # Generate more realistic features with correlations
    age = np.random.normal(35, 12, n_samples)
    age = np.clip(age, 18, 75).astype(int)
    
    # Income correlated with age and education
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
    credit_limit = income * 0.3  # Assume credit limit is 30% of income
    credit_utilization = debt / (credit_limit + 1)  # Avoid division by zero
    credit_utilization = np.clip(credit_utilization, 0, 1)
    
    # Add some additional realistic features
    credit_inquiries = np.random.poisson(2, n_samples)  # Number of credit inquiries
    credit_inquiries = np.clip(credit_inquiries, 0, 10)
    
    # Create target variable using realistic credit scoring logic
    # FICO-like scoring factors
    payment_factor = payment_history * 0.35  # 35% weight
    utilization_factor = (1 - credit_utilization) * 100 * 0.30  # 30% weight
    length_factor = np.minimum(employment_length * 10, 100) * 0.15  # 15% weight
    income_factor = np.minimum((income - 30000) / 1000, 50) * 0.10  # 10% weight
    inquiry_factor = (10 - credit_inquiries) * 10 * 0.10  # 10% weight
    
    credit_score = payment_factor + utilization_factor + length_factor + income_factor + inquiry_factor
    
    # Determine creditworthiness (threshold at 60% of max score)
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
    
    return data

def load_credit_data():
    """Load credit scoring data from file"""
    try:
        if os.path.exists('credit_data.csv'):
            data = pd.read_csv('credit_data.csv')
            print(f"Credit data loaded: {len(data)} samples")
            return data
        else:
            print("Credit data file not found, generating synthetic data...")
            return generate_credit_data()
    except Exception as e:
        print(f"Error loading credit data: {e}")
        print("Generating synthetic data instead...")
        return generate_credit_data()

def train_credit_model():
    """Train and evaluate credit scoring models"""
    print("Loading credit data...")
    data = load_credit_data()
    
    # Prepare features and target
    X = data.drop('creditworthy', axis=1)
    y = data['creditworthy']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'model': model,
            'scaler': scaler if name == 'Logistic Regression' else None,
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
    
    print(f"\nBest model: {best_model_name} (ROC-AUC: {results[best_model_name]['roc_auc']:.4f})")
    
    # Save best model
    model_data = {
        'model': best_model,
        'scaler': best_scaler,
        'feature_names': X.columns.tolist(),
        'model_name': best_model_name
    }
    
    with open('credit_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("Model saved as 'credit_model.pkl'")
    
    # Create visualizations
    create_visualizations(results, y_test, X.columns)
    
    return results

def create_visualizations(results, y_test, feature_names):
    """Create ROC curves and feature importance plots"""
    plt.figure(figsize=(15, 5))
    
    # ROC Curves
    plt.subplot(1, 3, 1)
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
        plt.plot(fpr, tpr, label=f'{name} (AUC = {result["roc_auc"]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True)
    
    # Feature Importance (for Random Forest)
    if 'Random Forest' in results:
        rf_model = results['Random Forest']['model']
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.subplot(1, 3, 2)
        plt.title('Feature Importance (Random Forest)')
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
    plt.savefig('result/credit_model_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    train_credit_model()
