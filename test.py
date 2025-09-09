import pandas as pd
import pickle

# Load dataset
data = pd.read_csv("data/Mall_Customers.csv")
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Load model and scaler
with open("models/kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Transform and predict
X_scaled = scaler.transform(X)
data['Cluster'] = kmeans.predict(X_scaled)

# Show sample results
print("Sample clustered data:\n")
print(data.head())

# Cluster summary
print("\nCluster Characteristics:\n")
print(data.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean())
