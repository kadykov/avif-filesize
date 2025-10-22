import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import pickle

# Load data
data = pd.read_csv('data/dataset.csv')

# Get low_res_bit_density for each image (resolution 400)
low_res_data = data[data['resolution'] == 400][['image_name', 'bit_density']].rename(columns={'bit_density': 'low_res_bit_density'})

# Merge low_res_bit_density back to data
data = data.merge(low_res_data, on='image_name', how='left')

# Filter to rows with resolution > 400
train_eval_data = data[data['resolution'] > 400].copy()

# Group by image_name and split unique image_names into train/test
unique_images = train_eval_data['image_name'].unique()
train_images, test_images = train_test_split(unique_images, test_size=0.2, random_state=42)

# Filter data based on image names
train_data = train_eval_data[train_eval_data['image_name'].isin(train_images)]
test_data = train_eval_data[train_eval_data['image_name'].isin(test_images)]

# Features and target
features = ['entropy', 'variance', 'edge_density', 'y_entropy', 'y_variance', 'u_variance', 'v_variance', 'y_edge_density', 'y_mean', 'laplacian_variance', 'gradient_magnitude', 'rms_contrast', 'low_res_bit_density']
target = 'bit_density'

X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

# Train Polynomial Regression (degree 1 with StandardScaler)
model = Pipeline([('scaler', StandardScaler()), ('poly', PolynomialFeatures(degree=1)), ('linear', LinearRegression())])
model.fit(X_train, y_train)

# Evaluate on test set
y_pred_bit_density = model.predict(X_test)
# Reconstruct file_size
predicted_file_size = y_pred_bit_density * test_data['num_pixels'] / 8
actual_file_size = test_data['file_size']
# Compute R2
r2 = r2_score(actual_file_size, predicted_file_size)

print(f"Alternative Model R2 Score (file_size): {r2}")

# Save model
with open('data/alternative_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Alternative model saved as data/alternative_model.pkl")