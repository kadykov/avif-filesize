# Project Report: AVIF File Size Analysis

## Objective

The objective of this project is to analyze the factors influencing AVIF file sizes, particularly how image properties such as resolution, entropy, variance, and edge density correlate with file size. Additionally, to build a predictive model for estimating AVIF file sizes based on these properties.

## Methodology

### Dataset Selection

The dataset consists of 100 images downloaded from the Unsplash API, specifically from the user 'kadykov'. Images are selected to provide a diverse set of photographic content.

### Parameters

- Resolutions: 400, 800, 1200, 1600, 2000 pixels (width), with height adjusted to maintain aspect ratio.
- AVIF encoding quality: 65 (using avifenc tool).

### Data Generation

For each image, the following steps are performed:
1. Download the original JPG image.
2. Resize the image to each of the specified resolutions.
3. Convert the resized image to AVIF format using avifenc with quality 65.
4. Record the file size of the resulting AVIF file.

### Metrics

For each image, the following metrics are computed from the grayscale version:
- Entropy: Shannon entropy of pixel intensities.
- Variance: Variance of pixel intensities.
- Edge Density: Fraction of pixels identified as edges using the Canny edge detection algorithm.

### Analysis

- Compute Pearson correlation coefficients between file_size and each metric, as well as between resolution and file_size.
- Generate scatter plots of file_size vs each metric, colored by resolution.

### Modeling

- Features: resolution, entropy, variance, edge_density.
- Target: file_size.
- Model: Linear Regression.
- Train/Test Split: 80/20 split based on unique images to avoid data leakage.
- Evaluation Metrics: Mean Squared Error (MSE), Mean Absolute Error (MAE), R-squared (R2) score.

## Results

### Correlations

The Pearson correlation coefficients are calculated as follows:
- file_size vs entropy
- file_size vs variance
- file_size vs edge_density
- resolution vs file_size

The scatter plots (available in data/plots/) show the relationships between file_size and the metrics, with resolution as a color dimension.

### Model Evaluation

We experimented with three regression models to predict AVIF file sizes: Linear Regression, Polynomial Regression (degree 2), and Random Forest Regressor. All models use the same features (resolution, entropy, variance, edge_density) and the same 80/20 train/test split based on unique images.

#### Model Performance Comparison

| Model                  | MSE              | MAE         | R² Score |
|------------------------|------------------|-------------|----------|
| Linear Regression     | 22,147,061,795.57 | 76,363.85 | 0.46    |
| Polynomial Regression | 14,681,401,743.39 | 54,228.74 | 0.64    |
| Random Forest         | 17,311,271,626.84 | 52,991.47 | 0.58    |

The Polynomial Regression model performed best with the lowest MSE and highest R² score. The best model is saved to `data/best_model.pkl`.

## Conclusions

The analysis reveals that resolution is the strongest predictor of AVIF file size, with image complexity metrics also contributing. Among the tested models, Polynomial Regression (degree 2) provides the best prediction performance with an R² score of 0.64, outperforming both Linear Regression and Random Forest models.

## Potential Extensions

- Experiment with different AVIF quality settings.
- Use more advanced models like Random Forest or Neural Networks.
- Include additional image features such as color histograms or texture descriptors.
- Expand the dataset with more diverse images.
- Investigate compression efficiency for different image types.

## Model Performance Visualizations

### Scatter Plot: Actual vs Predicted File Size

The scatter plot shows the relationship between actual and predicted file sizes, colored by resolution. Points closer to the diagonal line indicate better predictions. Higher resolutions tend to have larger file sizes and potentially larger prediction errors.

![Actual vs Predicted](data/plots/actual_vs_predicted.png)

### Box Plot: Relative Prediction Errors by Resolution

The box plot displays the distribution of relative prediction errors (as percentage) grouped by resolution. Relative errors provide a normalized view of prediction accuracy across different file sizes.

![Errors by Resolution](data/plots/errors_by_resolution.png)

These visualizations highlight that the model performs reasonably well across resolutions, with relative errors showing consistent performance patterns.
