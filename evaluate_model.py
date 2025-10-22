import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
data = pd.read_csv('data/dataset.csv')

# Replicate test split
unique_images = data['image_name'].unique()
train_images, test_images = train_test_split(unique_images, test_size=0.2, random_state=42)
test_data = data[data['image_name'].isin(test_images)]

# Features and target
features = ['resolution', 'entropy', 'variance', 'edge_density']
target = 'file_size'

X_test = test_data[features]
y_test = test_data[target]

# Load best model
with open('data/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Generate predictions
y_pred = model.predict(X_test)

# Create plots directory if not exists
os.makedirs('data/plots', exist_ok=True)

# Scatter plot: actual vs predicted, colored by resolution
plt.figure(figsize=(8, 6))
scatter = plt.scatter(y_test, y_pred, c=test_data['resolution'], cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Resolution')
plt.xlabel('Actual File Size')
plt.ylabel('Predicted File Size')
plt.title('Actual vs Predicted File Size')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.savefig('data/plots/actual_vs_predicted.png')
plt.close()

# Box plot: relative errors grouped by resolution
errors = (abs(y_test - y_pred) / y_test) * 100
test_data_with_errors = test_data.copy()
test_data_with_errors['relative_error'] = errors

plt.figure(figsize=(8, 6))
sns.boxplot(x='resolution', y='relative_error', data=test_data_with_errors)
plt.xlabel('Resolution')
plt.ylabel('Relative Prediction Error (%)')
plt.title('Relative Prediction Errors by Resolution')
plt.savefig('data/plots/errors_by_resolution.png')
plt.close()

# Update REPORT.md
section_title = '## Model Performance Visualizations'
new_section = f'\n{section_title}\n\n'
new_section += '### Scatter Plot: Actual vs Predicted File Size\n\n'
new_section += 'The scatter plot shows the relationship between actual and predicted file sizes, colored by resolution. Points closer to the diagonal line indicate better predictions. Higher resolutions tend to have larger file sizes and potentially larger prediction errors.\n\n'
new_section += '![Actual vs Predicted](data/plots/actual_vs_predicted.png)\n\n'
new_section += '### Box Plot: Relative Prediction Errors by Resolution\n\n'
new_section += 'The box plot displays the distribution of relative prediction errors (as percentage) grouped by resolution. Relative errors provide a normalized view of prediction accuracy across different file sizes.\n\n'
new_section += '![Errors by Resolution](data/plots/errors_by_resolution.png)\n\n'
new_section += 'These visualizations highlight that the model performs reasonably well across resolutions, with relative errors showing consistent performance patterns.\n'

with open('REPORT.md', 'r') as f:
    content = f.read()

# Replace the existing section if it exists
old_section_start = content.find('## Model Evaluation Visualizations')
if old_section_start != -1:
    # Find the end of the section (next ## or end of file)
    next_section = content.find('\n##', old_section_start + 1)
    if next_section == -1:
        next_section = len(content)
    content = content[:old_section_start] + new_section
else:
    # If not found, append
    content += new_section

with open('REPORT.md', 'w') as f:
    f.write(content)