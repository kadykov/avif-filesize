import os
import subprocess
import csv
import numpy as np
from PIL import Image
from skimage.measure import shannon_entropy
from skimage.feature import canny
from skimage.color import rgb2gray, rgb2ycbcr
from skimage.filters import sobel, laplace

widths = [400, 800, 1200, 1600, 2000]
images_dir = 'data/images'
avif_dir = 'data/avif'
os.makedirs(avif_dir, exist_ok=True)
csv_path = 'data/dataset.csv'

with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['image_name', 'resolution', 'file_size', 'entropy', 'variance', 'edge_density', 'y_entropy', 'y_variance', 'u_variance', 'v_variance', 'y_edge_density', 'y_mean', 'laplacian_variance', 'gradient_magnitude', 'rms_contrast', 'num_pixels', 'bit_density'])

    for filename in os.listdir(images_dir):
        if not filename.endswith('.jpg'):
            continue
        image_name = filename[:-4]  # remove .jpg
        img_path = os.path.join(images_dir, filename)
        img = Image.open(img_path)
        # compute metrics
        gray = rgb2gray(np.array(img))
        entropy = shannon_entropy(gray)
        variance = np.var(gray)
        edges = canny(gray)
        edge_density = np.sum(edges) / edges.size
        # YUV metrics
        ycbcr = rgb2ycbcr(np.array(img))
        Y = ycbcr[:, :, 0]
        U = ycbcr[:, :, 1]
        V = ycbcr[:, :, 2]
        y_entropy = shannon_entropy(Y)
        y_variance = np.var(Y)
        u_variance = np.var(U)
        v_variance = np.var(V)
        y_edges = canny(Y)
        y_edge_density = np.sum(y_edges) / y_edges.size
        y_mean = np.mean(Y)
        # New features
        laplacian_filtered = laplace(Y)
        laplacian_variance = np.var(laplacian_filtered)
        gradient_magnitude = np.mean(sobel(Y))
        rms_contrast = np.std(Y)
        for width in widths:
            height = int(width * img.height / img.width)
            num_pixels = width * height
            resized = img.resize((width, height), Image.LANCZOS)
            # save temp
            temp_path = f'temp_{image_name}_{width}.png'
            resized.save(temp_path)
            # convert to avif
            avif_name = f'{image_name}_{width}.avif'
            avif_path = os.path.join(avif_dir, avif_name)
            subprocess.run(['avifenc', temp_path, '-o', avif_path, '-q', '50'], check=True)
            # get size
            file_size = os.path.getsize(avif_path)
            bit_density = (file_size * 8) / num_pixels
            # write to csv
            writer.writerow([image_name, width, file_size, entropy, variance, edge_density, y_entropy, y_variance, u_variance, v_variance, y_edge_density, y_mean, laplacian_variance, gradient_magnitude, rms_contrast, num_pixels, bit_density])
            # remove temp
            os.remove(temp_path)