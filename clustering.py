import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from data_preprocessing import load_and_preprocess_data
import joblib

def apply_kmeans(X_scaled, n_clusters=4):
    """Apply K-means clustering."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    return kmeans, clusters

def cluster_customers():
    # Load data
    df, X_scaled, scaler = load_and_preprocess_data()
    
    # Apply K-means
    kmeans, clusters = apply_kmeans(X_scaled)
    df['cluster'] = clusters
    
    # Save results
    df.to_csv('data/customer_data_clustered.csv', index=False)
    joblib.dump(kmeans, 'kmeans_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Clustered data saved to 'data/customer_data_clustered.csv'")
    print("Model saved as 'kmeans_model.pkl', Scaler as 'scaler.pkl'")
    
    return df, kmeans, scaler

if __name__ == "__main__":
    df, kmeans, scaler = cluster_customers()
    print("Cluster Counts:\n", df['cluster'].value_counts())
