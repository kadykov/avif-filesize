import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

# Load data
data = pd.read_csv('data/dataset.csv')

# Filter to resolution >= 1200
data = data[data['resolution'] >= 1200]

# Group by image_name and split unique image_names into train/test
unique_images = data['image_name'].unique()
train_images, test_images = train_test_split(unique_images, test_size=0.2, random_state=42)

# Filter data based on image names
train_data = data[data['image_name'].isin(train_images)]
test_data = data[data['image_name'].isin(test_images)]

# Features and target
features = ['entropy', 'variance', 'edge_density', 'y_entropy', 'y_variance', 'u_variance', 'v_variance', 'y_edge_density', 'y_mean', 'laplacian_variance', 'gradient_magnitude', 'rms_contrast', 'num_pixels']
target = 'bit_density'

X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

# Train Polynomial Regression (degree 1) with StandardScaler
model = Pipeline([('scaler', StandardScaler()), ('poly', PolynomialFeatures(degree=1)), ('linear', LinearRegression())])
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Full High Res Polynomial Regression (degree 1 with StandardScaler):")
print(f"MSE: {mse}")
print(f"MAE: {mae}")
print(f"R2 Score: {r2}")

# Feature importance (coefficients from linear model)
poly_features = model.named_steps['poly'].get_feature_names_out(features)
coefficients = model.named_steps['linear'].coef_
feature_importance = dict(zip(poly_features, coefficients))
print("\nFeature Importance (Polynomial Coefficients):")
for feat, coef in sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True):
    print(f"{feat}: {coef}")

# Save model
with open('data/full_high_res_model.pkl', 'wb') as f:
    pickle.dump(model, f)