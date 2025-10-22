import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

# Load data
data = pd.read_csv('data/dataset.csv')

# Group by image_name and split unique image_names into train/test
unique_images = data['image_name'].unique()
train_images, test_images = train_test_split(unique_images, test_size=0.2, random_state=42)

# Filter data based on image names
train_data = data[data['image_name'].isin(train_images)]
test_data = data[data['image_name'].isin(test_images)]

# Features and target (exclude resolution)
features = ['entropy', 'variance', 'edge_density', 'y_entropy', 'y_variance', 'u_variance', 'v_variance', 'y_edge_density', 'y_mean']
target = 'bit_density'

X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

# Train Polynomial Regression (degree 2)
poly_model = Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression())])
poly_model.fit(X_train, y_train)

# Train Random Forest
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate both models
models = {'Polynomial Regression': poly_model, 'Random Forest': rf_model}
best_model = None
best_r2 = -float('inf')

for name, model in models.items():
    # Predict bit_density
    y_pred_bit_density = model.predict(X_test)
    # Compute predicted file_size
    predicted_file_size = y_pred_bit_density * test_data['num_pixels'] / 8
    actual_file_size = test_data['file_size']
    # Compute metrics for file_size
    mse = mean_squared_error(actual_file_size, predicted_file_size)
    mae = mean_absolute_error(actual_file_size, predicted_file_size)
    r2 = r2_score(actual_file_size, predicted_file_size)

    print(f"{name} (evaluating file_size prediction):")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"R2 Score: {r2}")
    print()

    if r2 > best_r2:
        best_r2 = r2
        best_model = model

# Save best model
with open('data/bit_density_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print(f"Best model saved: {type(best_model).__name__} with R2: {best_r2}")