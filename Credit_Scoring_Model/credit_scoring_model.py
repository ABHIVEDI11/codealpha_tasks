# CREDIT SCORING MODEL
# ===================
# This script implements a comprehensive credit scoring model that predicts
# an individual's creditworthiness using machine learning algorithms.
#
# Features:
# - Generates synthetic credit data with realistic financial features
# - Performs feature engineering to create additional predictive variables
# - Trains multiple models: Logistic Regression, Decision Trees, Random Forest
# - Evaluates models using precision, recall, F1-score, and ROC-AUC metrics
# - Creates visualizations for model comparison and feature importance
# - Saves all results to organized files in a 'result' folder
#
# Key Components:
# - CreditScoringModel class: Main class containing all model functionality
# - Data generation: Creates synthetic credit data with realistic distributions
# - Feature engineering: Builds interaction features and risk scores
# - Model training: Trains three different classification algorithms
# - Model evaluation: Comprehensive performance assessment
# - Visualization: ROC curves, confusion matrices, feature importance plots
# - Results export: Saves detailed results to text and CSV files
#
# Usage: Run this script to train and evaluate the credit scoring models
# Output: All results saved to 'result' folder with detailed metrics and plots

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os

# CREDIT SCORING MODEL CLASS
# =========================
# This class encapsulates all functionality for credit scoring model development
# including data generation, feature engineering, model training, evaluation,
# and results visualization.
#
# Attributes:
# - models: Dictionary to store trained machine learning models
# - scaler: StandardScaler object for feature normalization
# - results: Dictionary to store model evaluation results
#
# Methods:
# - generate_sample_data(): Creates synthetic credit data
# - feature_engineering(): Builds additional predictive features
# - prepare_data(): Splits and scales data for modeling
# - train_models(): Trains multiple classification algorithms
# - evaluate_models(): Assesses model performance
# - create_visualizations(): Generates performance plots
# - save_results(): Exports results to files

class CreditScoringModel:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.results = {}
        
    # GENERATE SAMPLE DATA METHOD
    # ===========================
    # Creates synthetic credit data with realistic distributions for demonstration
    # 
    # Parameters:
    # - n_samples: Number of data points to generate (default: 1000)
    #
    # Features generated:
    # - age: Normal distribution around 35 years
    # - income: Log-normal distribution for realistic income spread
    # - debt_to_income: Beta distribution for debt ratios
    # - payment_history: Beta distribution for payment reliability
    # - credit_utilization: Beta distribution for credit usage
    # - length_of_credit: Exponential distribution for credit history
    # - number_of_accounts: Poisson distribution for account count
    # - late_payments_30d/60d: Poisson distribution for payment delays
    # - inquiries_last_6m: Poisson distribution for credit inquiries
    #
    # Target variable: credit_worthy (binary classification)
    # Created using weighted combination of features

    def generate_sample_data(self, n_samples=1000):
        """Generate synthetic credit data for demonstration"""
        np.random.seed(42)
        
        data = {
            'age': np.random.normal(35, 10, n_samples).astype(int),
            'income': np.random.lognormal(10.5, 0.5, n_samples),
            'debt_to_income': np.random.beta(2, 5, n_samples) * 0.8,
            'payment_history': np.random.beta(3, 2, n_samples),
            'credit_utilization': np.random.beta(2, 3, n_samples),
            'length_of_credit': np.random.exponential(5, n_samples),
            'number_of_accounts': np.random.poisson(8, n_samples),
            'late_payments_30d': np.random.poisson(1, n_samples),
            'late_payments_60d': np.random.poisson(0.5, n_samples),
            'inquiries_last_6m': np.random.poisson(2, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create target variable based on features
        credit_score = (
            df['income'] * 0.3 +
            (1 - df['debt_to_income']) * 0.2 +
            df['payment_history'] * 0.25 +
            (1 - df['credit_utilization']) * 0.15 +
            df['length_of_credit'] * 0.1
        )
        
        df['credit_worthy'] = (credit_score > credit_score.median()).astype(int)
        
        return df
    
    # FEATURE ENGINEERING METHOD
    # ==========================
    # Creates additional predictive features from the original dataset
    # to improve model performance and capture complex relationships
    #
    # Parameters:
    # - df: Input DataFrame with original features
    #
    # New features created:
    # - income_debt_ratio: Ratio of income to debt-to-income ratio
    # - payment_credit_ratio: Ratio of payment history to credit utilization
    # - late_payment_score: Weighted combination of late payments
    # - inquiry_risk: Normalized credit inquiry risk
    # - age_category: Categorical age groups converted to numerical codes
    #
    # Returns: DataFrame with original + engineered features

    def feature_engineering(self, df):
        """Create additional features"""
        df_engineered = df.copy()
        
        # Create interaction features
        df_engineered['income_debt_ratio'] = df_engineered['income'] / (df_engineered['debt_to_income'] + 0.01)
        df_engineered['payment_credit_ratio'] = df_engineered['payment_history'] / (df_engineered['credit_utilization'] + 0.01)
        
        # Create risk scores
        df_engineered['late_payment_score'] = df_engineered['late_payments_30d'] + df_engineered['late_payments_60d'] * 2
        df_engineered['inquiry_risk'] = df_engineered['inquiries_last_6m'] / 10
        
        # Age categories
        df_engineered['age_category'] = pd.cut(df_engineered['age'], 
                                             bins=[0, 25, 35, 50, 100], 
                                             labels=['young', 'young_adult', 'adult', 'senior'])
        
        # Convert categorical to numerical
        df_engineered['age_category'] = df_engineered['age_category'].cat.codes
        
        return df_engineered
    
    # PREPARE DATA METHOD
    # ===================
    # Prepares the engineered dataset for machine learning modeling
    # by selecting features, splitting data, and scaling features
    #
    # Parameters:
    # - df: Input DataFrame with all features (original + engineered)
    #
    # Steps performed:
    # 1. Select relevant features for modeling
    # 2. Split data into training (80%) and test (20%) sets
    # 3. Scale features using StandardScaler for consistent model performance
    #
    # Returns: Scaled training/test data, labels, and feature column names

    def prepare_data(self, df):
        """Prepare data for modeling"""
        # Select features
        feature_columns = [
            'age', 'income', 'debt_to_income', 'payment_history', 'credit_utilization',
            'length_of_credit', 'number_of_accounts', 'late_payments_30d', 'late_payments_60d',
            'inquiries_last_6m', 'income_debt_ratio', 'payment_credit_ratio', 
            'late_payment_score', 'inquiry_risk', 'age_category'
        ]
        
        X = df[feature_columns]
        y = df['credit_worthy']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_columns
    
    # TRAIN MODELS METHOD
    # ===================
    # Trains multiple machine learning models for credit scoring
    # to compare performance across different algorithms
    #
    # Parameters:
    # - X_train: Scaled training features
    # - y_train: Training labels (credit_worthy)
    #
    # Models trained:
    # 1. Logistic Regression: Linear model with regularization
    # 2. Decision Tree: Non-linear model with interpretable rules
    # 3. Random Forest: Ensemble model with 100 trees
    #
    # All models use random_state=42 for reproducibility

    def train_models(self, X_train, y_train):
        """Train multiple models"""
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            self.models[name] = model
    
    # EVALUATE MODELS METHOD
    # ======================
    # Evaluates all trained models using comprehensive metrics
    # and stores results for comparison and visualization
    #
    # Parameters:
    # - X_test: Scaled test features
    # - y_test: Test labels (credit_worthy)
    #
    # Metrics calculated:
    # - AUC Score: Area Under ROC Curve for model discrimination
    # - Cross-validation Accuracy: 5-fold CV accuracy with standard deviation
    # - Classification Report: Precision, Recall, F1-score for each class
    # - Predictions and Probabilities: For visualization and analysis
    #
    # Results stored in self.results dictionary for each model

    def evaluate_models(self, X_test, y_test):
        """Evaluate all models and store results"""
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='accuracy')
            
            self.results[name] = {
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'auc_score': auc_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
    
    # CREATE VISUALIZATIONS METHOD
    # ============================
    # Generates comprehensive visualizations for model performance analysis
    # and saves them as high-quality PNG files in the result folder
    #
    # Parameters:
    # - X_test: Scaled test features
    # - y_test: Test labels (credit_worthy)
    #
    # Visualizations created:
    # 1. ROC Curves: Comparison of all models' discrimination ability
    # 2. Confusion Matrices: Detailed classification performance for each model
    # 3. Feature Importance: Top 10 most important features (Random Forest only)
    #
    # Files saved:
    # - roc_curves.png: ROC curve comparison plot
    # - confusion_matrices.png: Confusion matrix heatmaps
    # - feature_importance.png: Feature importance bar plot

    def create_visualizations(self, X_test, y_test):
        """Create and save visualizations"""
        if not os.path.exists('result'):
            os.makedirs('result')
        
        # ROC Curves
        plt.figure(figsize=(10, 6))
        for name, result in self.results.items():
            fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
            plt.plot(fpr, tpr, label=f'{name} (AUC = {result["auc_score"]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('result/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Confusion Matrices
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, (name, result) in enumerate(self.results.items()):
            cm = confusion_matrix(y_test, result['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{name} Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('result/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature Importance (for Random Forest)
        if 'Random Forest' in self.models:
            rf_model = self.models['Random Forest']
            feature_importance = pd.DataFrame({
                'feature': [f'Feature_{i}' for i in range(len(rf_model.feature_importances_))],
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
            plt.title('Top 10 Feature Importance (Random Forest)')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig('result/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # SAVE RESULTS METHOD
    # ===================
    # Exports all model evaluation results to organized files
    # for detailed analysis and reporting
    #
    # Parameters:
    # - y_test: Test labels (credit_worthy) for classification reports
    #
    # Files created:
    # 1. model_results.txt: Detailed text report with all metrics
    # 2. model_summary.csv: Summary table with key performance metrics
    #
    # Content includes:
    # - AUC scores for all models
    # - Cross-validation accuracy and standard deviation
    # - Detailed classification reports (precision, recall, F1-score)
    # - Summary table for easy comparison
    #
    # Returns: DataFrame with model performance summary

    def save_results(self, y_test):
        """Save results to files"""
        if not os.path.exists('result'):
            os.makedirs('result')
        
        # Save detailed results
        with open('result/model_results.txt', 'w') as f:
            f.write("CREDIT SCORING MODEL RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            for name, result in self.results.items():
                f.write(f"MODEL: {name}\n")
                f.write("-" * 30 + "\n")
                f.write(f"AUC Score: {result['auc_score']:.4f}\n")
                f.write(f"Cross-validation Accuracy: {result['cv_mean']:.4f} (+/- {result['cv_std']*2:.4f})\n\n")
                
                f.write("Classification Report:\n")
                f.write(classification_report(y_test, result['predictions']))
                f.write("\n" + "=" * 50 + "\n\n")
        
        # Save summary
        summary_data = []
        for name, result in self.results.items():
            summary_data.append({
                'Model': name,
                'AUC Score': f"{result['auc_score']:.4f}",
                'CV Accuracy': f"{result['cv_mean']:.4f}",
                'CV Std': f"{result['cv_std']:.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('result/model_summary.csv', index=False)
        
        print("Results saved to 'result' folder")
        return summary_df

# MAIN FUNCTION
# =============
# Orchestrates the complete credit scoring model pipeline
# from data generation to results export
#
# Pipeline steps:
# 1. Initialize CreditScoringModel class
# 2. Generate synthetic credit data (1000 samples)
# 3. Perform feature engineering to create additional features
# 4. Prepare data by splitting and scaling
# 5. Train three different machine learning models
# 6. Evaluate models using comprehensive metrics
# 7. Create visualizations for performance analysis
# 8. Save all results to organized files
# 9. Display performance summary
#
# Output: Complete credit scoring model with detailed results
# saved to 'result' folder for analysis and reporting

def main():
    print("Credit Scoring Model - Training and Evaluation")
    print("=" * 50)
    
    # Initialize model
    model = CreditScoringModel()
    
    # Generate data
    print("Generating sample credit data...")
    df = model.generate_sample_data(1000)
    print(f"Generated {len(df)} samples")
    
    # Feature engineering
    print("Performing feature engineering...")
    df_engineered = model.feature_engineering(df)
    print(f"Created {len(df_engineered.columns)} features")
    
    # Prepare data
    print("Preparing data for modeling...")
    X_train, X_test, y_train, y_test, feature_columns = model.prepare_data(df_engineered)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train models
    print("Training models...")
    model.train_models(X_train, y_train)
    print("Models trained successfully")
    
    # Evaluate models
    print("Evaluating models...")
    model.evaluate_models(X_test, y_test)
    
    # Create visualizations
    print("Creating visualizations...")
    model.create_visualizations(X_test, y_test)
    
    # Save results
    print("Saving results...")
    summary = model.save_results(y_test)
    
    # Display summary
    print("\nMODEL PERFORMANCE SUMMARY")
    print("=" * 30)
    print(summary.to_string(index=False))
    
    print(f"\nAll results saved to 'result' folder")
    print("Files created:")
    print("- result/model_results.txt (detailed results)")
    print("- result/model_summary.csv (summary table)")
    print("- result/roc_curves.png (ROC curves)")
    print("- result/confusion_matrices.png (confusion matrices)")
    print("- result/feature_importance.png (feature importance)")

if __name__ == "__main__":
    main()
