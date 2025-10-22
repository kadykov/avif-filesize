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

The linear regression model is trained and evaluated on the test set, with metrics printed to the console.

## Conclusions

The analysis reveals that resolution is the strongest predictor of AVIF file size, with image complexity metrics also contributing. The linear model provides a prediction of file sizes based on the features.

## Potential Extensions

- Experiment with different AVIF quality settings.
- Use more advanced models like Random Forest or Neural Networks.
- Include additional image features such as color histograms or texture descriptors.
- Expand the dataset with more diverse images.
- Investigate compression efficiency for different image types.