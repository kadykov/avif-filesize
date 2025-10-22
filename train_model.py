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

# Features and target
features = ['resolution', 'entropy', 'variance', 'edge_density']
target = 'file_size'

X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate Linear Regression
y_pred_lr = model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("Linear Regression:")
print(f"MSE: {mse_lr}")
print(f"MAE: {mae_lr}")
print(f"R2 Score: {r2_lr}")

# Polynomial Regression (degree 2)
model_poly = Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression())])
model_poly.fit(X_train, y_train)

y_pred_poly = model_poly.predict(X_test)
mse_poly = mean_squared_error(y_test, y_pred_poly)
mae_poly = mean_absolute_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print("\nPolynomial Regression (degree 2):")
print(f"MSE: {mse_poly}")
print(f"MAE: {mae_poly}")
print(f"R2 Score: {r2_poly}")

# Random Forest Regressor
model_rf = RandomForestRegressor(random_state=42)
model_rf.fit(X_train, y_train)

y_pred_rf = model_rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\nRandom Forest Regressor:")
print(f"MSE: {mse_rf}")
print(f"MAE: {mae_rf}")
print(f"R2 Score: {r2_rf}")

# Compare models (select best based on lowest MSE)
models = {
    'Linear Regression': (mse_lr, mae_lr, r2_lr, model),
    'Polynomial Regression': (mse_poly, mae_poly, r2_poly, model_poly),
    'Random Forest': (mse_rf, mae_rf, r2_rf, model_rf)
}

best_model_name = min(models, key=lambda x: models[x][0])
best_mse, best_mae, best_r2, best_model = models[best_model_name]

print(f"\nBest model: {best_model_name}")
print(f"MSE: {best_mse}")
print(f"MAE: {best_mae}")
print(f"R2 Score: {best_r2}")

# Save best model
with open('data/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Also save the original linear model
with open('data/model.pkl', 'wb') as f:
    pickle.dump(model, f)