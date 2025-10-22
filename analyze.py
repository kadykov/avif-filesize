import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the dataset
df = pd.read_csv('data/dataset.csv')

# Compute Pearson correlations
correlations = {
    'file_size vs entropy': df['file_size'].corr(df['entropy']),
    'file_size vs variance': df['file_size'].corr(df['variance']),
    'file_size vs edge_density': df['file_size'].corr(df['edge_density']),
    'resolution vs file_size': df['resolution'].corr(df['file_size'])
}

# Create plots directory if not exists
os.makedirs('data/plots', exist_ok=True)

# Generate scatter plots
metrics = ['entropy', 'variance', 'edge_density']
for metric in metrics:
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(df['file_size'], df[metric], c=df['resolution'], cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Resolution')
    plt.xlabel('File Size')
    plt.ylabel(metric.capitalize())
    plt.title(f'File Size vs {metric.capitalize()}')
    plt.savefig(f'data/plots/file_size_vs_{metric}.png')
    plt.close()

# Output summary
print("Pearson Correlation Coefficients:")
for key, value in correlations.items():
    print(f"{key}: {value:.4f}")