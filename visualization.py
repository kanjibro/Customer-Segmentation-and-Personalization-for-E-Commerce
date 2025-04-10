import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from clustering import cluster_customers

def visualize_segments():
    # Load clustered data
    df, _, _ = cluster_customers()
    
    # Scatter plot: Age vs Total Spent
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='age', y='total_spent', hue='cluster', palette='deep')
    plt.title('Customer Segments: Age vs Total Spent')
    plt.xlabel('Age')
    plt.ylabel('Total Spent ($)')
    plt.legend(title='Cluster')
    plt.savefig('segments_age_spent.png')
    plt.close()
    
    # Box plot: Annual Income by Cluster
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='cluster', y='annual_income', palette='deep')
    plt.title('Annual Income Distribution by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Annual Income ($)')
    plt.savefig('income_by_cluster.png')
    plt.close()
    
    print("Visualizations saved as 'segments_age_spent.png' and 'income_by_cluster.png'")

if __name__ == "__main__":
    visualize_segments()
