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
# Get coefficients
coef = model.named_steps['linear'].coef_
feature_importance = list(zip(features, coef))
# Sort by absolute coefficient
sorted_features = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)
print("Feature coefficients:")
for feat, coeff in sorted_features:
    print(f"{feat}: {coeff}")
# Select top 5, including low_res_bit_density
top_features = [f for f, c in sorted_features[:5]]
if 'low_res_bit_density' not in top_features:
    top_features = top_features[:4] + ['low_res_bit_density']
print(f"Selected top features: {top_features}")

# Retrain with selected features
X_train_simp = train_data[top_features]
X_test_simp = test_data[top_features]
model_simp = Pipeline([('scaler', StandardScaler()), ('poly', PolynomialFeatures(degree=1)), ('linear', LinearRegression())])
model_simp.fit(X_train_simp, y_train)

# Evaluate simplified model
y_pred_simp = model_simp.predict(X_test_simp)
predicted_file_size_simp = y_pred_simp * test_data['num_pixels'] / 8
actual_file_size = test_data['file_size']
r2_simp = r2_score(actual_file_size, predicted_file_size_simp)
print(f"Simplified Model R2 Score (file_size): {r2_simp}")

# Feature importance for simplified model
coef_simp = model_simp.named_steps['linear'].coef_
simp_importance = list(zip(top_features, coef_simp))
print("Simplified model feature importance:")
for feat, coeff in simp_importance:
    print(f"{feat}: {coeff}")

# Save simplified model
with open('data/simplified_model.pkl', 'wb') as f:
    pickle.dump(model_simp, f)
print("Simplified model saved as data/simplified_model.pkl")


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