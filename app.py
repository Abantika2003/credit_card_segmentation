import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
# ============================================
# GLOBAL MATPLOTLIB / SEABORN SETTINGS
# ============================================
sns.set_style("whitegrid")

plt.rcParams.update({
    "figure.dpi": 100,          # clarity
    "savefig.dpi": 100,
    "font.size": 8,             # base font size
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.figsize": (4, 3),   # default small figure size
})

# ============================================
# STREAMLIT APP - CREDIT CARD SEGMENTATION
# ============================================

st.set_page_config(page_title="Credit Card Usage Segmentation", layout="wide")

st.title("💡 Credit Card Behavior Analytics Dashboard")
st.markdown("Explore customer credit card usage through intelligent segmentation and gain insights that drive smarter financial strategies.")


# -------------------------------
# 1️⃣ Upload Dataset
# -------------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
else:
    data = pd.read_csv("data/Customer Data.csv")

st.subheader("📄 First 10 Rows of Data")
st.dataframe(data.head(10))

# Drop customer ID if present
if "CUST_ID" in data.columns:
    data = data.drop("CUST_ID", axis=1)

# Fill missing values
data = data.fillna(data.mean())

# -------------------------------
# 2️⃣ Exploratory Data Analysis (EDA)
# -------------------------------
st.subheader("🔍 Exploratory Data Analysis")

# Correlation Heatmap
st.markdown("**Correlation Heatmap**")
fig, ax = plt.subplots(figsize=(4, 3))
sns.heatmap(data.corr(), cmap="coolwarm", annot=False, ax=ax)
st.pyplot(fig)
plt.close(fig)

# Histograms
st.markdown("**Feature Distributions**")
num_features = ["BALANCE", "PURCHASES", "CASH_ADVANCE", "PAYMENTS"]
for col in num_features:
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.histplot(data[col], bins=30, kde=True, color="skyblue", ax=ax)
    ax.set_title(f"Distribution of {col}", fontsize=9)
    st.pyplot(fig)
    plt.close(fig)


# -------------------------------
# 3️⃣ Preprocessing & PCA
# -------------------------------
st.subheader("⚙️ Preprocessing & PCA")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

st.write("✅ PCA complete. Reduced data shape:", X_pca.shape)

# PCA variance
fig, ax = plt.subplots(figsize=(4, 3))
ax.bar(["PC1", "PC2"], pca.explained_variance_ratio_, color="teal")
ax.set_title("PCA Explained Variance Ratio", fontsize=9)
st.pyplot(fig)
plt.close(fig)


# -------------------------------
# 4️⃣ Clustering
# -------------------------------
st.subheader("🤖 Clustering Models")

# KMeans
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(X_pca)

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_pca)

# Show cluster visualizations
st.markdown("**KMeans Clusters (2D PCA Projection)**")
fig, ax = plt.subplots(figsize=(4, 3))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_kmeans, palette="Set2", s=15, ax=ax)
ax.set_title("KMeans Clusters", fontsize=9)
ax.legend(fontsize=6, loc='best')
st.pyplot(fig)
plt.close(fig)

st.markdown("**DBSCAN Clusters (2D PCA Projection)**")
fig, ax = plt.subplots(figsize=(4, 3))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_dbscan, palette="tab10", s=15, ax=ax)
ax.set_title("DBSCAN Clusters", fontsize=9)
ax.legend(fontsize=6, loc='best')
st.pyplot(fig)
plt.close(fig)


# -------------------------------
# 5️⃣ Evaluation
# -------------------------------
st.subheader("📊 Cluster Evaluation Metrics")

def evaluate_model(name, X, labels):
    if len(set(labels)) <= 1:
        return {"Model": name, "Silhouette": None, "Davies-Bouldin": None, "Calinski-Harabasz": None}
    return {
        "Model": name,
        "Silhouette": silhouette_score(X, labels),
        "Davies-Bouldin": davies_bouldin_score(X, labels),
        "Calinski-Harabasz": calinski_harabasz_score(X, labels)
    }

results = [
    evaluate_model("KMeans", X_pca, labels_kmeans),
    evaluate_model("DBSCAN", X_pca, labels_dbscan)
]

results_df = pd.DataFrame(results)
st.dataframe(results_df)

# -------------------------------
# 6️⃣ Business Insights
# -------------------------------

st.subheader("💼 Advanced Business Intelligence & Insights")

# Attach cluster labels to data
clustered_data = data.copy()
clustered_data["Cluster"] = labels_kmeans

# 1️⃣ Summary of Each Cluster (Key Metrics)
cluster_summary = clustered_data.groupby("Cluster").agg({
    "BALANCE": "mean",
    "PURCHASES": "mean",
    "ONEOFF_PURCHASES": "mean" if "ONEOFF_PURCHASES" in clustered_data.columns else "mean",
    "INSTALLMENTS_PURCHASES": "mean" if "INSTALLMENTS_PURCHASES" in clustered_data.columns else "mean",
    "CASH_ADVANCE": "mean",
    "CREDIT_LIMIT": "mean" if "CREDIT_LIMIT" in clustered_data.columns else "mean",
    "PAYMENTS": "mean",
    "MINIMUM_PAYMENTS": "mean" if "MINIMUM_PAYMENTS" in clustered_data.columns else "mean",
    "PRC_FULL_PAYMENT": "mean" if "PRC_FULL_PAYMENT" in clustered_data.columns else "mean"
}).round(2)

st.markdown("### 📊 Cluster Summary Statistics")
st.dataframe(cluster_summary)

# 2️⃣ Label Each Cluster (Manual business meaning)
cluster_labels = {
    0: "💳 Premium Customers (High spenders, pay full balance)",
    1: "💸 Cash Advance Users (Frequent withdrawals, low full payments)",
    2: "🛍️ Regular Customers (Moderate usage and spending)",
    3: "💤 Dormant Users (Low usage, small payments)"
}
clustered_data["Cluster_Label"] = clustered_data["Cluster"].map(cluster_labels)

# 3️⃣ Visualize Spending Patterns
st.markdown("### 💰 Average Spending & Payment per Cluster")
fig, ax = plt.subplots(figsize=(5, 3))
cluster_summary[["PURCHASES", "CASH_ADVANCE", "PAYMENTS"]].plot(kind="bar", ax=ax)
ax.set_title("Average Purchase, Cash Advance, and Payment per Cluster", fontsize=9)
ax.set_ylabel("Average Amount")
ax.tick_params(axis='x', rotation=0)
st.pyplot(fig)
plt.close(fig)

# 4️⃣ Full Payment Behavior
st.markdown("### 📈 Full Payment Ratio per Cluster")
if "PRC_FULL_PAYMENT" in cluster_summary.columns:
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.barplot(x=cluster_summary.index, y=cluster_summary["PRC_FULL_PAYMENT"], palette="coolwarm", ax=ax)
    ax.set_title("Full Payment Ratio by Cluster", fontsize=9)
    ax.set_ylabel("Average % of Full Payment")
    st.pyplot(fig)
    plt.close(fig)

# 5️⃣ Cluster Distribution
st.markdown("### 🧩 Customer Distribution by Cluster")
fig, ax = plt.subplots(figsize=(4, 4))
cluster_counts = clustered_data["Cluster"].value_counts()
ax.pie(cluster_counts, labels=[f"Cluster {i}" for i in cluster_counts.index],
       autopct="%1.1f%%", startangle=90, colors=sns.color_palette("Set2"), textprops={'fontsize': 8})
ax.set_title("Customer Distribution Across Clusters", fontsize=9)
st.pyplot(fig)
plt.close(fig)


# 6️⃣ Business Recommendations Table
st.markdown("### 💼 Strategic Business Recommendations")
recommendations = pd.DataFrame({
    "Cluster": [f"Cluster {i}" for i in cluster_labels.keys()],
    "Description": list(cluster_labels.values()),
    "Suggested Action": [
        "Offer premium rewards, higher credit limits",
        "Promote EMI plans, control credit risk",
        "Provide loyalty offers to maintain engagement",
        "Send reactivation or cashback offers"
    ]
})
st.dataframe(recommendations)


# -------------------------------
# 7️⃣ AI Predictor (Improved Input Experience)
# -------------------------------
st.subheader("🤖 Predict Cluster for New Customer")

# Train simple AI model (RandomForest)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, labels_kmeans, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save model
os.makedirs("data", exist_ok=True)
joblib.dump(rf_model, "data/ai_cluster_predictor.pkl")

# User input section
st.markdown("### ✍️ Enter Customer Details")
st.markdown("Type numeric values for each feature below (press Enter after typing).")

# Create a form to hold inputs neatly
with st.form("user_input_form"):
    user_input = {}
    cols = st.columns(2)  # two columns for cleaner layout

    for i, col_name in enumerate(data.columns):
        with cols[i % 2]:
            val = st.text_input(
                f"{col_name}",
                value=str(round(data[col_name].mean(), 2))
            )
            # Convert safely to float (handles blank or invalid input)
            try:
                user_input[col_name] = float(val)
            except ValueError:
                user_input[col_name] = float(data[col_name].mean())

    submitted = st.form_submit_button("🔍 Predict Cluster")

if submitted:
    user_df = pd.DataFrame([user_input])
    user_scaled = scaler.transform(user_df)
    predicted_cluster = rf_model.predict(user_scaled)[0]
    st.success(f"🎯 Predicted Cluster: {predicted_cluster}")
