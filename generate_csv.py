import pandas as pd
import numpy as np

def generate_csv(n_samples=100, output_file='data/customer_data.csv'):
    np.random.seed(42)  # For reproducibility
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
    df.to_csv(output_file, index=False)
    print(f"Generated {output_file} with {n_samples} rows.")

if __name__ == "__main__":
    # Ensure data/ folder exists
    import os
    if not os.path.exists('data'):
        os.makedirs('data')
    generate_csv()
