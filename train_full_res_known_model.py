import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import joblib

# Load dataset
df = pd.read_csv('data/dataset.csv')

# Group by image_name to find max resolution
df['max_res'] = df.groupby('image_name')['resolution'].transform('max')

# Create df for training: rows where resolution >= 1200 and < max_res
df_train = df[(df['resolution'] >= 1200) & (df['resolution'] < df['max_res'])].copy()

# Add full_res_bit_density to df_train
def get_full_res_bd(row):
    return df[(df['image_name'] == row['image_name']) & (df['resolution'] == row['max_res'])]['bit_density'].iloc[0]

df_train['full_res_bit_density'] = df_train.apply(get_full_res_bd, axis=1)

# Features and target
features = ['entropy', 'variance', 'edge_density', 'y_entropy', 'y_variance', 'u_variance', 'v_variance', 'y_edge_density', 'y_mean', 'laplacian_variance', 'gradient_magnitude', 'rms_contrast', 'full_res_bit_density']
target = 'bit_density'

# Split images into train and test
unique_images = df['image_name'].unique()
np.random.seed(42)  # for reproducibility
np.random.shuffle(unique_images)
split_idx = len(unique_images) // 2
train_images = unique_images[:split_idx]
test_images = unique_images[split_idx:]

# Train df
train_df = df_train[df_train['image_name'].isin(train_images)]

# Test df: all rows for test images
test_df = df[df['image_name'].isin(test_images)].copy()
test_df['max_res'] = test_df.groupby('image_name')['resolution'].transform('max')
test_df['full_res_bit_density'] = test_df.apply(get_full_res_bd, axis=1)

# Prepare X and y
X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)

# Reconstruct file_size
predicted_file_size = y_pred * test_df['num_pixels'] / 8
actual_file_size = test_df['file_size']

# Compute R2
r2 = r2_score(actual_file_size, predicted_file_size)

print(f'R2 Score: {r2:.4f}')

# Save model and scaler
joblib.dump({'model': model, 'scaler': scaler}, 'data/full_res_known_model_v2.pkl')