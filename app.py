import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load model and scaler
with open("models/kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load dataset
data = pd.read_csv("data/Mall_Customers.csv")

st.title("🛍️ Customer Segmentation App")
st.write("Clustering customers using KMeans based on Annual Income and Spending Score.")

# Scale and predict
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]
X_scaled = scaler.transform(X)
data['Cluster'] = kmeans.predict(X_scaled)

# Show cluster summary
st.subheader("Cluster Characteristics")
st.write(data.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean())

# Plot clusters
fig, ax = plt.subplots()
sns.scatterplot(
    x="Annual Income (k$)", y="Spending Score (1-100)",
    hue="Cluster", palette="Set2", s=100, data=data, ax=ax
)
st.pyplot(fig)
