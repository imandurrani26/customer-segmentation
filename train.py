import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle

# ✅ Make sure models folder exists
os.makedirs("models", exist_ok=True)

# ✅ Make sure data folder exists
os.makedirs("data", exist_ok=True)

# Load dataset
data_path = "data/Mall_Customers.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ Dataset not found at {data_path}")

data = pd.read_csv(data_path)

# Select features
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KMeans
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_scaled)

# Save model and scaler inside models/
with open("models/kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Training complete! Model and scaler saved inside 'models/'")
