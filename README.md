# 🏥 Automated Medical Risk Stratification System
### *An Interactive Framework for Clustering & Classification of Health Data*

This repository contains a comprehensive Machine Learning tool designed to analyze patient health profiles and categorize them into risk groups using Unsupervised and Supervised learning techniques.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Mathematical Framework](#mathematical-framework)
- [Clustering Methodologies](#clustering-methodologies)
- [System Architecture](#system-architecture)
- [Installation & Setup](#installation--setup)
- [Performance Metrics](#performance-metrics)

---

## 🔍 Overview
The core objective of this project is to simulate a medical diagnostic environment where clinical features like **Age**, **BMI**, **Blood Pressure**, and **Cholesterol** are used to identify patterns in patient health. The system allows users to interactively select features and compare how different algorithms partition the data.

---

## 🧪 Mathematical Framework
To ensure scientific accuracy, the synthetic data is generated using a weighted health-risk scoring system:
$$RiskScore = \frac{Age}{80} + \frac{BMI}{30} + \frac{SystolicBP}{140} + \frac{Cholesterol}{240}$$

The system evaluates clustering quality using two primary metrics:
1. **WSS (Within-cluster Sum of Squares):** Measures internal consistency (compactness).
2. **BSS (Between-cluster Sum of Squares):** Measures separation between different health groups.

---

## 🤖 Clustering Methodologies
The application implements three distinct clustering paradigms for comparative analysis:

### 1. K-Means (Partitioning)
A centroid-based approach that minimizes the variance within clusters. Fast and efficient for large medical datasets.

### 2. Agglomerative (Hierarchical - Bottom Up)
Starts with individual patients and merges them into groups based on Euclidean distance. Ideal for discovering fine-grained sub-categories of diseases.

### 3. DIANA (Hierarchical - Top Down)
Divisive Analysis starts with the entire population as one cluster and iteratively splits it. This is highly effective in medical triage to isolate high-risk outliers from the general population.

---

## 🚀 System Architecture
- **Backend:** Python 3.13
- **Frontend:** Streamlit (Interactive UI)
- **Visualization:** - **Plotly:** For interactive 3D spatial analysis.
  - **Seaborn/Matplotlib:** For 1D/2D distribution plots.
- **Preprocessing:** Scikit-Learn `StandardScaler` to ensure features with different units (e.g., Age vs. BP) contribute equally.

---

## 🛠 Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Sinaaraste/Medical-Clustering-System.git

   cd Medical-Clustering-System


2.  **Create a Virtual Environment:**
```bash
    python -m venv ml_env    # On Linux/macOS
    source ml_env/bin/activate
  


    python -m venv ml_env     # On Windows
    .\ml_env\Scripts\activate  
```

3. **Install Dependencies:**
```bash
    pip install -r requirements.txt
```
4. **Launch the Application:**
```bash
    streamlit run index.py
```