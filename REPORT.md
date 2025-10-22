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

Additionally, metrics from the YUV color space:
- Y Entropy: Shannon entropy of the Y (luminance) channel.
- Y Variance: Variance of the Y channel.
- U Variance: Variance of the U (chrominance) channel.
- V Variance: Variance of the V (chrominance) channel.
- Y Edge Density: Fraction of pixels identified as edges in the Y channel using Canny.
- Y Mean: Mean value of the Y channel.
- Laplacian Variance: Variance of the Laplacian filter applied to the Y channel.
- Gradient Magnitude: Mean of the Sobel gradient magnitude on the Y channel.
- RMS Contrast: Standard deviation of the Y channel (root mean square contrast).

### Analysis

- Compute Pearson correlation coefficients between file_size and each metric, as well as between resolution and file_size.
- Generate scatter plots of file_size vs each metric, colored by resolution.

### Modeling

- Features: resolution, entropy, variance, edge_density, y_entropy, y_variance, u_variance, v_variance, y_edge_density, y_mean.
- Target: file_size.
- Model: Polynomial Regression (degree 2).
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

We trained a Polynomial Regression (degree 2) model using all features including the new YUV metrics to predict AVIF file sizes. The model uses an 80/20 train/test split based on unique images.

#### Enhanced Model Performance

| Metric | Value |
|--------|-------|
| MSE    | 14,105,664,883.29 |
| MAE    | 62,761.48 |
| R² Score | 0.66 |

The enhanced model achieves an R² score of 0.66, showing improved performance with the additional features. The model is saved to `data/enhanced_model.pkl`.

### Bit Density Approach

To explore an alternative modeling strategy, we introduced a bit density metric, calculated as (file_size * 8) / num_pixels, representing the average bits per pixel for each AVIF file. This approach aims to decouple the resolution dependency by predicting bit density from image complexity metrics, then reconstructing file size as bit_density * num_pixels / 8.

- Features: entropy, variance, edge_density, y_entropy, y_variance, u_variance, v_variance, y_edge_density, y_mean (excluding resolution).
- Target: bit_density.
- Models: Polynomial Regression (degree 1 with StandardScaler) and Random Forest.
- Evaluation: Predict bit_density on test set, compute predicted file_size = bit_density * num_pixels / 8, then evaluate MSE, MAE, and R² for file_size prediction.

#### Bit Density Model Performance

| Model | MSE | MAE | R² Score |
|-------|-----|-----|----------|
| Polynomial Regression | 16,148,562,144.73 | 78,046.53 | 0.61 |
| Random Forest | 10,285,801,320.81 | 52,615.29 | 0.75 |

The Random Forest model performed best with an R² score of 0.75 for file_size prediction. The best model is saved to `data/bit_density_model.pkl`.

### Final Model with Additional Features

To further improve the model, we added three additional features computed on the Y channel: Laplacian Variance, Gradient Magnitude, and RMS Contrast.

- Features: entropy, variance, edge_density, y_entropy, y_variance, u_variance, v_variance, y_edge_density, y_mean, laplacian_variance, gradient_magnitude, rms_contrast.
- Target: bit_density.
- Models: Polynomial Regression (degree 2) and Random Forest.
- Evaluation: Predict bit_density on test set, compute predicted file_size = bit_density * num_pixels / 8, then evaluate MSE, MAE, and R² for file_size prediction.

#### Final Model Performance

| Model | MSE | MAE | R² Score |
|-------|-----|-----|----------|
| Polynomial Regression (degree 1 with scaling) | 4,545,842,022.29 | 41,145.08 | 0.89 |
| Random Forest | 6,672,626,856.07 | 50,376.14 | 0.84 |

The Polynomial Regression model (degree 1 with StandardScaler) achieved an R² score of 0.89 for file_size prediction, outperforming the Random Forest's 0.84. The final model is saved to `data/final_model.pkl`.


## Conclusions

The analysis reveals that resolution and image complexity metrics, including those derived from YUV color space, are important predictors of AVIF file size. The enhanced Polynomial Regression (degree 2) model achieves an R² score of 0.66, demonstrating improved predictive performance with the inclusion of additional YUV-based features. Feature importance analysis highlights the significance of interactions between variance, entropy, and edge density metrics.

The bit density approach, using Random Forest to predict bits per pixel from complexity metrics (excluding resolution), achieves an R² score of 0.75 for file_size prediction, outperforming the direct file_size prediction models. By incorporating additional features such as Laplacian variance, gradient magnitude, and RMS contrast, the final Polynomial Regression model (degree 1 with StandardScaler) achieves an R² score of 0.89. This suggests that modeling compression efficiency separately from resolution, with comprehensive image complexity features, is a highly effective strategy for predicting AVIF file sizes.

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
