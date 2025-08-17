#!/usr/bin/env python3
"""
Disease Prediction Pipeline - Example Usage

This script demonstrates how to use the disease prediction pipeline
with different datasets and configurations. It shows practical examples
of training models for heart disease and diabetes prediction.
"""

import subprocess
import sys
import os

# =============================================================================
# COMMAND EXECUTION UTILITY FUNCTION
# =============================================================================
# This function executes shell commands and displays results:
# 1. Takes a command string and description as input
# 2. Executes the command using subprocess
# 3. Captures both stdout and stderr output
# 4. Displays success or error messages appropriately
# 5. Provides clear feedback to the user
def run_command(command, description):
    """Run a command and display the result."""
    print(f"{description}")
    print("-" * 70)
    
    try:
        # Execute the command and capture output
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        # Check if command executed successfully
        if result.returncode == 0:
            print("Success! Check the output directory for results.")
        else:
            print(f"Error: {result.stderr}")
            
    except Exception as e:
        print(f"Error running command: {e}")
    
    print()

# =============================================================================
# MAIN FUNCTION - DEMONSTRATES PIPELINE USAGE
# =============================================================================
# This function demonstrates the complete workflow by:
# 1. Running heart disease prediction with advanced options (SMOTE + calibration)
# 2. Running diabetes prediction with SMOTE for imbalanced data
# 3. Providing clear explanations of what each command does
# 4. Summarizing the results and next steps
def main():
    """Main function to demonstrate usage."""
    print("Disease Prediction Pipeline - Example Usage")
    print("=" * 70)
    
    # Example 1: Heart Disease Prediction with Advanced Features
    # This demonstrates using SMOTE for imbalanced data and probability calibration
    print("1. Heart Disease Prediction")
    print("-" * 70)
    run_command(
        "python disease_prediction_pipeline.py --data heart.csv --target target --outdir runs/heart --smote --calibrate",
        "Running: python disease_prediction_pipeline.py --data heart.csv --target target --outdir runs/heart --smote --calibrate"
    )
    
    # Example 2: Diabetes Prediction with SMOTE
    # This shows how to handle imbalanced datasets using SMOTE technique
    print("2. Diabetes Prediction")
    print("-" * 70)
    run_command(
        "python disease_prediction_pipeline.py --data diabetes.csv --target Outcome --outdir runs/diabetes --smote",
        "Running: python disease_prediction_pipeline.py --data diabetes.csv --target Outcome --outdir runs/diabetes --smote"
    )
    
    # Provide comprehensive summary of what was accomplished
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("Both examples completed successfully!")
    
    # Guide users to where they can find the results
    print("\nCheck the following directories for results:")
    print("- runs/heart/")
    print("- runs/diabetes/")
    
    # Explain what each output directory contains
    print("\nEach directory contains:")
    print("- Model comparison metrics")
    print("- Visualization plots")
    print("- Best trained model (pickle file)")
    print("- Feature importance analysis")

if __name__ == "__main__":
    main()
