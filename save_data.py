import numpy as np
import pandas as pd

def generate_and_save():
    n = 1000  # Number of patients
    np.random.seed(42)
    
    # 1. Generate Raw Features
    age = np.random.normal(50, 12, n).clip(20, 80)
    bmi = np.random.normal(27, 5, n).clip(15, 45)
    bp = 80 + (0.5 * age) + (1.2 * bmi) + np.random.normal(0, 8, n)
    chol = 140 + (0.4 * age) + (1.5 * bmi) + np.random.normal(0, 12, n)
    
    # 2. Calculate Risk Score and Labels
    score = (age/80) + (bmi/30) + (bp/140) + (chol/240)
    y = np.where(score < 2.3, 0, 1)
    y = np.where(score > 2.9, 2, y)
    
    # 3. Create DataFrame
    df = pd.DataFrame({
        'Age': age,
        'BMI': bmi,
        'Blood_Pressure': bp,
        'Cholesterol': chol,
        'Risk_Score': score,
    })
    
    # 4. Save to CSV
    df.to_csv('medical_dataset.csv', index=False)
    print("✅ File 'medical_dataset.csv' has been created successfully!")

if __name__ == "__main__":
    generate_and_save()