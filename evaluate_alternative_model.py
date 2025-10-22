import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
data = pd.read_csv('data/dataset.csv')

# Get low_res_bit_density for each image (resolution 400)
low_res_data = data[data['resolution'] == 400][['image_name', 'bit_density']].rename(columns={'bit_density': 'low_res_bit_density'})

# Merge low_res_bit_density back to data
data = data.merge(low_res_data, on='image_name', how='left')

# Filter to rows with resolution > 400
eval_data = data[data['resolution'] > 400].copy()

# Group by image_name and split unique image_names into train/test (same as training)
unique_images = eval_data['image_name'].unique()
train_images, test_images = train_test_split(unique_images, test_size=0.2, random_state=42)

# Filter data based on image names
test_data = eval_data[eval_data['image_name'].isin(test_images)]

# Features
features = ['entropy', 'variance', 'edge_density', 'y_entropy', 'y_variance', 'u_variance', 'v_variance', 'y_edge_density', 'y_mean', 'laplacian_variance', 'gradient_magnitude', 'rms_contrast', 'low_res_bit_density']

X_test = test_data[features]
y_test_bit_density = test_data['bit_density']
actual_file_size = test_data['file_size']

# Load alternative model
with open('data/alternative_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Generate predictions
y_pred_bit_density = model.predict(X_test)

# Reconstruct file_size from predicted bit_density
predicted_file_size = y_pred_bit_density * test_data['num_pixels'] / 8

# Compute relative errors
relative_errors = (abs(actual_file_size - predicted_file_size) / actual_file_size) * 100

# Create plots directory if not exists
os.makedirs('data/plots', exist_ok=True)

# Scatter plot: actual vs predicted file_size, colored by resolution
plt.figure(figsize=(8, 6))
scatter = plt.scatter(actual_file_size, predicted_file_size, c=test_data['resolution'], cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Resolution')
plt.xlabel('Actual File Size')
plt.ylabel('Predicted File Size')
plt.title('Actual vs Predicted File Size (Alternative Model)')
plt.plot([actual_file_size.min(), actual_file_size.max()], [actual_file_size.min(), actual_file_size.max()], 'k--', lw=2)
plt.savefig('data/plots/alternative_model_performance.png')
plt.close()

# Box plot: relative errors by resolution
test_data_with_errors = test_data.copy()
test_data_with_errors['relative_error'] = relative_errors

plt.figure(figsize=(8, 6))
sns.boxplot(x='resolution', y='relative_error', data=test_data_with_errors)
plt.xlabel('Resolution')
plt.ylabel('Relative Prediction Error (%)')
plt.title('Relative Prediction Errors by Resolution (Alternative Model)')
plt.savefig('data/plots/alternative_model_errors.png')
plt.close()

print("Evaluation complete. Plots saved to data/plots/alternative_model_performance.png and data/plots/alternative_model_errors.png")