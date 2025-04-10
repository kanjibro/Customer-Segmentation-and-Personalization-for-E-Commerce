import pandas as pd
import joblib

def personalize_offers(customer_id):
    """Sample personalization based on cluster."""
    # Load clustered data and model
    df = pd.read_csv('data/customer_data_clustered.csv')
    kmeans = joblib.load('kmeans_model.pkl')
    
    # Find customer
    customer = df[df['customer_id'] == customer_id]
    if customer.empty:
        return "Customer not found."
    
    cluster = customer['cluster'].values[0]
    
    # Simple personalization logic
    offers = {
        0: "10% off on next purchase - Low Spenders",
        1: "Free shipping on orders over $50 - Frequent Buyers",
        2: "Exclusive VIP discount - High Spenders",
        3: "Welcome bundle for new customers - Young Shoppers"
    }
    return f"Customer {customer_id} (Cluster {cluster}): {offers.get(cluster, 'General offer')}"

if __name__ == "__main__":
    for cid in [1, 2, 3, 4]:
        print(personalize_offers(cid))
