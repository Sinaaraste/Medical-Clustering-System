import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import rand_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy.cluster.hierarchy import linkage, fcluster

# -----------------------------
# DIANA Implementation
# -----------------------------
def diana(X, n_clusters):
    Z = linkage(X, method='ward')
    labels = fcluster(Z, t=n_clusters, criterion='maxclust') - 1
    return labels

# -----------------------------
# Utility Functions
# -----------------------------
def compute_uss(X, labels):
    uss = 0
    for label in np.unique(labels):
        cluster = X[labels == label]
        center = cluster.mean(axis=0)
        uss += ((cluster - center) ** 2).sum()
    return uss

def compute_bss(X, labels):
    overall_mean = X.mean(axis=0)
    bss = 0
    for label in np.unique(labels):
        cluster = X[labels == label]
        center = cluster.mean(axis=0)
        bss += len(cluster) * ((center - overall_mean) ** 2).sum()
    return bss

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Medical Clustering & Classification System (Final Version)")
st.write("4 Clustering Algorithms | Multiple k | USS/BSS/Rand | Classification | Analysis")

# Sidebar Settings
st.sidebar.header("Dataset Settings")
n_samples = st.sidebar.slider("Number of Samples", 100, 1000, 500)
real_k = st.sidebar.slider("True Number of Classes", 2, 5, 3)

st.sidebar.header("Visualization Settings")
selected_algo = st.sidebar.selectbox("Select Algorithm to Visualize", ["K-Means", "Agglomerative", "DIANA", "Ward Hierarchical"])
visual_k = st.sidebar.slider("Select k for Clustering Visualization", 2, 6, 3)

# -----------------------------
# Data Generation
# -----------------------------
X, y_true = make_blobs(
    n_samples=n_samples,
    centers=real_k,
    n_features=2,
    cluster_std=1.5,
    random_state=42
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Clustering for multiple k
# -----------------------------
k_values = range(2, 7)
algorithms = ["K-Means", "Agglomerative", "DIANA", "Ward Hierarchical"]

results = []
all_labels = {}  # برای نمودار همه kها و الگوریتم‌ها

for algo_name in algorithms:
    all_labels[algo_name] = {}
    for k in k_values:
        if algo_name == "K-Means":
            model = KMeans(n_clusters=k, random_state=42)
            labels = model.fit_predict(X_scaled)
        elif algo_name == "Agglomerative":
            model = AgglomerativeClustering(n_clusters=k, linkage="ward")
            labels = model.fit_predict(X_scaled)
        elif algo_name == "DIANA":
            labels = diana(X_scaled, k)
        elif algo_name == "Ward Hierarchical":
            model = AgglomerativeClustering(n_clusters=k, linkage="ward")
            labels = model.fit_predict(X_scaled)

        uss = compute_uss(X_scaled, labels)
        bss = compute_bss(X_scaled, labels)
        rand = rand_score(y_true, labels)

        results.append({
            "Algorithm": algo_name,
            "k": k,
            "USS": round(uss, 2),
            "BSS": round(bss, 2),
            "Rand Index": round(rand, 4)
        })

        all_labels[algo_name][k] = labels

results_df = pd.DataFrame(results)

st.subheader("Clustering Evaluation Results (All Algorithms & k)")
st.dataframe(results_df)

# -----------------------------
# Comparison Plots
# -----------------------------
st.subheader("Comparison of Metrics Across Algorithms and k")
metrics = ["USS", "BSS", "Rand Index"]

for metric in metrics:
    st.write(f"### {metric}")
    fig, ax = plt.subplots()
    for algo_name in algorithms:
        algo_data = results_df[results_df["Algorithm"] == algo_name]
        ax.plot(algo_data["k"], algo_data[metric], marker='o', label=algo_name)
    ax.set_xlabel("k")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs k for Different Algorithms")
    ax.legend()
    st.pyplot(fig)

# -----------------------------
# Clustering Visualization
# -----------------------------
st.subheader(f"Clustering Visualization ({selected_algo}, k={visual_k})")
fig, ax = plt.subplots()
ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=all_labels[selected_algo][visual_k], cmap='viridis')
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_title(f"{selected_algo} Clustering (k={visual_k})")
st.pyplot(fig)

# -----------------------------
# Classification Section
# -----------------------------
st.subheader("Classification (KNN)")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_true, test_size=0.3, random_state=42
)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

st.write("Classification Accuracy:", round(acc, 4))

fig2, ax2 = plt.subplots()
cax = ax2.matshow(cm, cmap='Blues')
for (i, j), val in np.ndenumerate(cm):
    ax2.text(j, i, int(val), ha='center', va='center', color='red')
fig2.colorbar(cax)
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")
st.pyplot(fig2)

# -----------------------------
# Analysis and Summary
# -----------------------------
st.subheader("Analysis and Summary")

st.write("این بخش تحلیل عددی نتایج خوشه‌بندی و طبقه‌بندی را نشان می‌دهد:")

best_uss_algo = results_df.loc[results_df["USS"].idxmin()]
st.write(f"- الگوریتمی با کمترین USS (بهترین خوشه‌بندی داخلی): **{best_uss_algo['Algorithm']}** با k={best_uss_algo['k']}, USS={best_uss_algo['USS']}")

best_bss_algo = results_df.loc[results_df["BSS"].idxmax()]
st.write(f"- الگوریتمی با بیشترین BSS (بهترین جداسازی خوشه‌ها): **{best_bss_algo['Algorithm']}** با k={best_bss_algo['k']}, BSS={best_bss_algo['BSS']}")

best_rand_algo = results_df.loc[results_df["Rand Index"].idxmax()]
st.write(f"- الگوریتمی با بالاترین Rand Index (بهترین مطابقت با برچسب واقعی): **{best_rand_algo['Algorithm']}** با k={best_rand_algo['k']}, Rand Index={best_rand_algo['Rand Index']}")

st.write(f"- الگوریتم طبقه‌بندی KNN روی همان داده‌ها دقت **{round(acc, 4)}** داشت. Confusion Matrix نشان می‌دهد که چند نمونه اشتباه دسته‌بندی شدند.")
