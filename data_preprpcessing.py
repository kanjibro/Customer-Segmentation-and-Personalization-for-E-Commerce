import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def generate_sample_data(n_samples=100):
    """Generate synthetic e-commerce customer data."""
    np.random.seed(42)
    customer_ids = range(1, n_samples + 1)
    ages = np.random.randint(18, 70, n_samples)
    annual_incomes = np.random.randint(20000, 100000, n_samples)
    purchase_frequencies = np.random.randint(1, 15, n_samples)
    total_spent = purchase_frequencies * np.random.uniform(10, 50, n_samples) + np.random.normal(0, 20, n_samples)
    
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'age': ages,
        'annual_income': annual_incomes,
        'purchase_frequency': purchase_frequencies,
        'total_spent': total_spent
    })
    df.to_csv('data/customer_data.csv', index=False)
    return df

def load_and_preprocess_data(file_path='data/customer_data.csv'):
    """Load and preprocess customer data."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Data file not found. Generating sample data...")
        df = generate_sample_data()
    
    # Handle missing values
    df.fillna(df.mean(), inplace=True)
    
    # Features for clustering
    features = ['age', 'annual_income', 'purchase_frequency', 'total_spent']
    X = df[features]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return df, X_scaled, scaler

if __name__ == "__main__":
    df, X_scaled, scaler = load_and_preprocess_data()
    print("Sample Data:\n", df.head())
    print("Scaled Features Shape:", X_scaled.shape)
