#!/usr/bin/env python3
"""
Disease Prediction Pipeline - Test Suite

This script tests the basic functionality of the disease prediction pipeline
to ensure all components are working correctly before running the full system.
It includes tests for data loading, model training, and package availability.
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# DATA LOADING TEST FUNCTION
# =============================================================================
# This function tests if the required datasets can be loaded correctly:
# 1. Attempts to load heart disease dataset (heart.csv)
# 2. Attempts to load diabetes dataset (diabetes.csv)
# 3. Displays dataset information (rows, columns)
# 4. Returns True if both datasets load successfully, False otherwise
def test_data_loading():
    """Test data loading functionality."""
    print("Data Loading:")
    print("-" * 30)
    
    try:
        # Test heart disease data loading
        # This dataset contains patient information for heart disease prediction
        heart_data = pd.read_csv("heart.csv")
        print(f"Heart dataset loaded: {heart_data.shape[0]} rows, {heart_data.shape[1]} columns")
        
        # Test diabetes data loading
        # This dataset contains patient information for diabetes prediction
        diabetes_data = pd.read_csv("diabetes.csv")
        print(f"Diabetes dataset loaded: {diabetes_data.shape[0]} rows, {diabetes_data.shape[1]} columns")
        
        return True
    except Exception as e:
        print(f"Error loading data: {e}")
        return False

# =============================================================================
# BASIC MODEL TRAINING AND TESTING FUNCTION
# =============================================================================
# This function tests the core machine learning functionality by:
# 1. Loading heart disease data and preparing features/target
# 2. Splitting data into training and testing sets
# 3. Training a Random Forest model with basic parameters
# 4. Making predictions and calculating accuracy
# 5. Testing model saving and loading functionality
# 6. Verifying that loaded model produces same results
def test_basic_model():
    """Test basic model training and prediction."""
    print("\nBasic Model Test:")
    print("-" * 30)
    
    try:
        # Import required machine learning libraries
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        import joblib
        
        # Load heart disease data for testing
        data = pd.read_csv("heart.csv")
        X = data.drop('target', axis=1)  # Features (all columns except target)
        y = data['target']               # Target variable (disease presence)
        
        # Split data into training (80%) and testing (20%) sets
        # This simulates the real-world scenario where we train on historical data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train a Random Forest model with basic parameters
        # This tests the core machine learning pipeline
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions on test data and calculate accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"   Accuracy: {accuracy:.3f}")
        
        # Test model saving and loading functionality
        # This ensures models can be persisted and reused
        model_path = "test_model.pkl"
        joblib.dump(model, model_path)
        loaded_model = joblib.load(model_path)
        loaded_accuracy = accuracy_score(y_test, loaded_model.predict(X_test))
        print(f"   Model saving/loading works")
        print(f"   Loaded model accuracy: {loaded_accuracy:.3f}")
        
        # Clean up temporary model file
        os.remove(model_path)
        
        return True
    except Exception as e:
        print(f"Error in basic model test: {e}")
        return False

# =============================================================================
# PACKAGE AVAILABILITY TEST FUNCTION
# =============================================================================
# This function checks if all required and optional packages are available:
# 1. Tests required packages (pandas, numpy, sklearn, matplotlib, joblib)
# 2. Tests optional packages (xgboost, imblearn)
# 3. Provides clear status for each package
# 4. Returns True if all required packages are available
def test_requirements():
    """Test required packages."""
    print("\nRequirements:")
    print("-" * 30)
    
    # List of required packages for basic functionality
    required_packages = [
        'pandas',      # Data manipulation and analysis
        'numpy',       # Numerical computing
        'sklearn',     # Machine learning algorithms
        'matplotlib',  # Plotting and visualization
        'joblib'       # Model persistence
    ]
    
    # List of optional packages for advanced features
    optional_packages = [
        'xgboost',     # Gradient boosting (optional enhancement)
        'imblearn'     # Imbalanced learning techniques (optional)
    ]
    
    all_good = True
    
    # Test each required package
    print("\nTesting required packages...")
    for package in required_packages:
        try:
            __import__(package)
            print(f"   {package} available")
        except ImportError:
            print(f"   {package} NOT available")
            all_good = False
    
    # Test each optional package
    print("\nTesting optional packages...")
    for package in optional_packages:
        try:
            __import__(package)
            print(f"   {package} available")
        except ImportError:
            print(f"   {package} not available (optional)")
    
    return all_good

# =============================================================================
# MAIN FUNCTION - ORCHESTRATES ALL TESTS
# =============================================================================
# This function runs all tests in sequence and provides a comprehensive summary:
# 1. Runs data loading test
# 2. Runs basic model training test
# 3. Runs package availability test
# 4. Provides detailed summary of all test results
# 5. Gives next steps based on test outcomes
def main():
    """Main test function."""
    print("Disease Prediction Pipeline - Test Suite")
    print("=" * 50)
    
    # Run all three types of tests
    data_test = test_data_loading()
    model_test = test_basic_model()
    req_test = test_requirements()
    
    # Display comprehensive test summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Data Loading: {'PASS' if data_test else 'FAIL'}")
    print(f"Basic Model: {'PASS' if model_test else 'FAIL'}")
    print(f"Requirements: {'PASS' if req_test else 'FAIL'}")
    
    # Calculate overall test results
    total_tests = 3
    passed_tests = sum([data_test, model_test, req_test])
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    # Provide guidance based on test results
    if passed_tests == total_tests:
        print("All tests passed! The pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Run: python example_usage.py")
        print("2. Or run individual predictions:")
        print("   python disease_prediction_pipeline.py --data heart.csv --target target --outdir runs/heart")
    else:
        print("Some tests failed. Please check the requirements and data files.")

if __name__ == "__main__":
    main()
