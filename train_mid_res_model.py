import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import pickle

# Load data
data = pd.read_csv('data/dataset.csv')

# Filter to mid-resolution: resolution >= 800 and <= 2000
data = data[(data['resolution'] >= 800) & (data['resolution'] <= 2000)]

# Group by image_name and split unique image_names into train/test
unique_images = data['image_name'].unique()
train_images, test_images = train_test_split(unique_images, test_size=0.2, random_state=42)

# Filter data based on image names
train_data = data[data['image_name'].isin(train_images)]
test_data = data[data['image_name'].isin(test_images)]

# Features: all metrics except low_res_bit_density (assuming it's not present, use all available metrics)
features = ['entropy', 'variance', 'edge_density', 'y_entropy', 'y_variance', 'u_variance', 'v_variance', 'y_edge_density', 'y_mean', 'laplacian_variance', 'gradient_magnitude', 'rms_contrast']
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
predicted_file_size = y_pred_bit_density * test_data['num_pixels'] / 8
actual_file_size = test_data['file_size']
r2 = r2_score(actual_file_size, predicted_file_size)

print(f"Mid-resolution model R2 Score for file_size prediction: {r2}")

# Save model
with open('data/mid_res_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as data/mid_res_model.pkl")