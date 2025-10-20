"""
Machine Learning Assignment 2 - Regression Analysis
Author: Divya Kapoor
Course: Machine Learning
Date: October 2024

NOTE: This code is for academic purposes only. 
Data files are not included to maintain academic integrity.
"""

import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
import os

def part1_linear_regression_demo():
    """Demo version of Part 1 - uses generated sample data instead of actual assignment data"""
    print("=== PART 1: Linear Regression Demo ===")
    
    # Generate sample data for demonstration (NOT the actual assignment data)
    np.random.seed(42)
    X_demo = np.random.rand(100, 1) * 10
    y_demo = 2.5 * X_demo.flatten() + 1.5 + np.random.randn(100) * 2
    
    # Convert to DataFrame
    data = pd.DataFrame({'X': X_demo.flatten(), 'y': y_demo})
    X = data['X'].values.reshape(-1, 1)
    y = data['y']
    
    # Plot demo data
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.7)
    plt.xlabel('Feature X (Demo Data)')
    plt.ylabel('Target y (Demo Data)')
    plt.title('Demo: Linear Regression - Synthetic Data\n(Actual assignment data not shown)')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Linear regression
    reg = linear_model.LinearRegression()
    reg.fit(X, y)
    y_pred = reg.predict(X)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.7, label='Demo Data')
    plt.plot(X, y_pred, 'r-', linewidth=2, label='Regression Line')
    plt.xlabel('Feature X (Demo Data)')
    plt.ylabel('Target y (Demo Data)')
    plt.title('Demo: Linear Regression Fit\n(Actual assignment data not shown)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("Demo model parameters:")
    print(f"Intercept (b0): {reg.intercept_:.4f}")
    print(f"Coefficient (b1): {reg.coef_[0]:.4f}")
    print("\nNote: This demonstrates the concept using synthetic data.")
    print("Actual assignment data is not included to maintain academic integrity.\n")

def part2_logistic_regression_demo():
    """Demo version of Part 2 - uses generated sample data"""
    print("=== PART 2: Logistic Regression Demo ===")
    
    # Generate sample data for demonstration
    np.random.seed(42)
    n_samples = 100
    Score1 = np.random.randn(n_samples) * 2 + 5
    Score2 = np.random.randn(n_samples) * 2 + 5
    y_demo = (Score1 + Score2 + np.random.randn(n_samples) > 10).astype(int)
    
    data = pd.DataFrame({'Score1': Score1, 'Score2': Score2, 'y': y_demo})
    X = data[['Score1', 'Score2']]
    y = data['y']
    
    # Plot demo data
    plt.figure(figsize=(10, 6))
    colors = ['red', 'blue']
    markers = ['o', 's']
    
    for i in range(2):
        mask = data['y'] == i
        plt.scatter(data['Score1'][mask], data['Score2'][mask], 
                   c=colors[i], marker=markers[i], 
                   label=f'Class {i}', alpha=0.7)
    
    plt.xlabel('Score 1 (Demo Data)')
    plt.ylabel('Score 2 (Demo Data)')
    plt.title('Demo: Logistic Regression - Synthetic Data\n(Actual assignment data not shown)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Logistic regression
    regS = linear_model.LogisticRegression()
    regS.fit(X, y)
    y_pred = regS.predict(X)
    
    accuracy = (y_pred == y).mean()
    print(f"Demo model accuracy: {accuracy:.2f}")
    print("Note: This demonstrates the concept using synthetic data.\n")

def part3_explanations():
    """Part 3: Explanations only (no code changes needed)"""
    print("=== PART 3: Multi-class Classification Explanations ===")
    
    # Your existing explanations here (keep them as is)
    print("One-vs-Rest (OvR) Method Explanation...")
    # [Your existing OvR explanation]
    
    print("\nOne-vs-One (OvO) Method Explanation...")
    # [Your existing OvO explanation]

def check_data_files():
    """Check if data files exist and warn about academic integrity"""
    data_files = ['./RegressionData.csv', './LogisticRegressionData.csv']
    
    for file in data_files:
        if os.path.exists(file):
            print(f"⚠️  WARNING: {file} found in directory.")
            print("   For academic integrity, do NOT commit actual assignment data to GitHub.")
            return False
    return True

def main():
    print("Machine Learning Assignment 2 - Demonstration Version")
    print("=" * 50)
    print("This version uses synthetic data to demonstrate concepts.")
    print("Actual assignment data is excluded to maintain academic integrity.\n")
    
    # Check for data files
    if not check_data_files():
        print("Proceeding with demo data instead of actual assignment data.\n")
    
    # Run demonstrations
    part1_linear_regression_demo()
    part2_logistic_regression_demo()
    part3_explanations()
    
    print("Demo completed successfully!")
    print("Note: This code demonstrates the concepts without compromising academic integrity.")

if __name__ == "__main__":
    main()