import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
data = pd.read_csv('data/dataset.csv')

# Filter to resolution >= 1200
data = data[data['resolution'] >= 1200]

# Group by image_name and split unique image_names into train/test
unique_images = data['image_name'].unique()
train_images, test_images = train_test_split(unique_images, test_size=0.2, random_state=42)

# Filter data based on image names
test_data = data[data['image_name'].isin(test_images)]

# Features and target for bit_density
features = ['entropy', 'variance', 'edge_density', 'y_entropy', 'y_variance', 'u_variance', 'v_variance', 'y_edge_density', 'y_mean', 'laplacian_variance', 'gradient_magnitude', 'rms_contrast', 'num_pixels']
target = 'bit_density'

X_test = test_data[features]
y_test_bit_density = test_data[target]

# Load full_high_res_model
with open('data/full_high_res_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Generate predictions for bit_density
y_pred_bit_density = model.predict(X_test)

# Reconstruct predicted file_size
predicted_file_size = y_pred_bit_density * test_data['num_pixels'] / 8
actual_file_size = test_data['file_size']

# Create plots directory if not exists
os.makedirs('data/plots', exist_ok=True)

# Scatter plot: actual vs predicted file_size, colored by resolution
plt.figure(figsize=(8, 6))
scatter = plt.scatter(actual_file_size, predicted_file_size, c=test_data['resolution'], cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Resolution')
plt.xlabel('Actual File Size')
plt.ylabel('Predicted File Size')
plt.title('Actual vs Predicted File Size (Full High Res Model)')
plt.plot([actual_file_size.min(), actual_file_size.max()], [actual_file_size.min(), actual_file_size.max()], 'k--', lw=2)
plt.savefig('data/plots/full_high_res_model_performance.png')
plt.close()

# Box plot: relative errors grouped by resolution
errors = (abs(actual_file_size - predicted_file_size) / actual_file_size) * 100
test_data_with_errors = test_data.copy()
test_data_with_errors['relative_error'] = errors

plt.figure(figsize=(8, 6))
sns.boxplot(x='resolution', y='relative_error', data=test_data_with_errors)
plt.xlabel('Resolution')
plt.ylabel('Relative Prediction Error (%)')
plt.title('Relative Prediction Errors by Resolution (Full High Res Model)')
plt.savefig('data/plots/full_high_res_model_errors.png')
plt.close()