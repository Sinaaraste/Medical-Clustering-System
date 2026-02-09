import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import rand_score
from sklearn.preprocessing import StandardScaler

# --- Page Configuration ---
st.set_page_config(page_title="Medical AI Analysis", layout="wide")
st.title("🏥 Medical Diagnosis & Clustering System")

# --- Section 1: Data Loading (Auto-detecting record count) ---
st.header("1. Data Configuration")

file_path = 'medical_dataset.csv'

if os.path.exists(file_path):
    df_raw = pd.read_csv(file_path)
    total_samples = len(df_raw)
    st.success(f"File loaded successfully. Total records: {total_samples}")
else:
    st.error(f"Error: '{file_path}' not found. Please ensure the CSV file exists.")
    st.stop()

# User only inputs K value
k_value = st.number_input("Enter K (Number of Clusters):", min_value=2, max_value=15, value=3)

st.write("---")
st.subheader("Select Features for Analysis:")

# Feature names matching your CSV columns
feature_names = ["Age", "BMI", "Blood_Pressure", "Cholesterol"]
selected_features = []
cols = st.columns(4)
for i, feature in enumerate(feature_names):
    if cols[i].checkbox(feature, value=True):
        selected_features.append(feature)

# --- DIANA Implementation ---
def run_diana(data, k):
    if k <= 1: return np.zeros(data.shape[0], dtype=int)
    labels = np.zeros(data.shape[0], dtype=int)
    n_c = 1
    while n_c < k:
        max_sse, to_split = -1, -1
        for c in range(n_c):
            c_data = data[labels == c]
            if len(c_data) <= 1: continue
            sse = np.sum((c_data - np.mean(c_data, axis=0))**2)
            if sse > max_sse:
                max_sse, to_split = sse, c
        if to_split == -1: break
        m = KMeans(n_clusters=2, n_init=10, random_state=42).fit(data[labels == to_split])
        labels[labels == to_split] = np.where(m.labels_ == 0, to_split, n_c)
        n_c += 1
    return labels

# --- Section 2: Main Execution Logic ---
if len(selected_features) > 0:
    # Extracting raw data for processing
    X_raw_all = df_raw[feature_names].values
    
    # Reconstructing ground truth labels from Risk_Score for Rand Index
    scores = df_raw['Risk_Score'].values
    y_true = np.where(scores < 2.3, 0, 1)
    y_true = np.where(scores > 2.9, 2, y_true)

    indices = [feature_names.index(f) for f in selected_features]
    X_selected = X_raw_all[:, indices]
    
    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    st.write("---")
    algo_choice = st.selectbox("Choose Clustering Algorithm:", 
                               ["K-Means", "Agglomerative (Bottom-Up)", "DIANA (Top-Down)"])

    if algo_choice == "K-Means":
        labels = KMeans(n_clusters=k_value, n_init=10, random_state=42).fit_predict(X_scaled)
    elif "Agglomerative" in algo_choice:
        labels = AgglomerativeClustering(n_clusters=k_value).fit_predict(X_scaled)
    else:
        labels = run_diana(X_scaled, k_value)

    # --- Metrics Visualization ---
    st.header(f"📊 Result Metrics (k={k_value})")
    global_mean = np.mean(X_scaled, axis=0)
    final_wss = sum(np.sum((X_scaled[labels == i] - np.mean(X_scaled[labels == i], axis=0))**2) for i in np.unique(labels))
    final_bss = sum(len(X_scaled[labels == i]) * np.sum((np.mean(X_scaled[labels == i], axis=0) - global_mean)**2) for i in np.unique(labels))
    
    m1, m2, m3 = st.columns(3)
    m1.metric("WSS", round(final_wss, 2))
    m2.metric("BSS", round(final_bss, 2))
    m3.metric("Rand Index (Accuracy)", round(rand_score(y_true, labels), 4))

    # --- Plotting Logic ---
    n_f = len(selected_features)
    st.header("🖼 Visual Analysis")
    if n_f == 1:
        fig, ax = plt.subplots(figsize=(10, 3))
        sns.stripplot(x=X_selected[:, 0], hue=labels, palette="viridis", ax=ax)
        ax.set_xlabel(selected_features[0])
        st.pyplot(fig)
    elif n_f == 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=X_selected[:, 0], y=X_selected[:, 1], hue=labels, palette="viridis", ax=ax)
        ax.set_xlabel(selected_features[0])
        ax.set_ylabel(selected_features[1])
        st.pyplot(fig)
    elif n_f == 3:
        fig_3d = px.scatter_3d(
            x=X_selected[:, 0], y=X_selected[:, 1], z=X_selected[:, 2], 
            color=labels.astype(str),
            labels={'x': selected_features[0], 'y': selected_features[1], 'z': selected_features[2]},
            title="Interactive 3D View"
        )
        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        st.info("High-dimensional data selected. 3D visualization is disabled for more than 3 features.")

    # --- Section 3: Performance Trends ---
    st.divider()
    st.header("📈 Performance Trends (k=1 to 10)")
    
    k_range = range(1, 11)
    wss_list, bss_list, ri_list = [], [], []

    with st.spinner('Calculating metrics...'):
        for i in k_range:
            if algo_choice == "K-Means":
                lab_k = KMeans(n_clusters=i, n_init=10, random_state=42).fit_predict(X_scaled)
            elif "Agglomerative" in algo_choice:
                lab_k = AgglomerativeClustering(n_clusters=i).fit_predict(X_scaled) if i > 1 else np.zeros(X_scaled.shape[0])
            else:
                lab_k = run_diana(X_scaled, i)
            
            w = sum(np.sum((X_scaled[lab_k == c] - np.mean(X_scaled[lab_k == c], axis=0))**2) for c in np.unique(lab_k))
            b = sum(len(X_scaled[lab_k == c]) * np.sum((np.mean(X_scaled[lab_k == c], axis=0) - global_mean)**2) for c in np.unique(lab_k))
            wss_list.append(w); bss_list.append(b); ri_list.append(rand_score(y_true, lab_k))

    c1, c2, c3 = st.columns(3)
    c1.subheader("WSS Trend")
    c1.line_chart(pd.DataFrame(wss_list, index=k_range, columns=["WSS"]))
    c2.subheader("BSS Trend")
    c2.line_chart(pd.DataFrame(bss_list, index=k_range, columns=["BSS"]))
    c3.subheader("Rand Index Trend")
    c3.line_chart(pd.DataFrame(ri_list, index=k_range, columns=["Rand Index"]))

else:
    st.error("Please select at least one feature to begin analysis.")