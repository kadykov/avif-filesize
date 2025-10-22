import os
import subprocess
import csv
import numpy as np
from PIL import Image
from skimage.measure import shannon_entropy
from skimage.feature import canny
from skimage.color import rgb2gray

widths = [400, 800, 1200, 1600, 2000]
images_dir = 'data/images'
avif_dir = 'data/avif'
os.makedirs(avif_dir, exist_ok=True)
csv_path = 'data/dataset.csv'

with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['image_name', 'resolution', 'file_size', 'entropy', 'variance', 'edge_density'])

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
        for width in widths:
            height = int(width * img.height / img.width)
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
            # write to csv
            writer.writerow([image_name, width, file_size, entropy, variance, edge_density])
            # remove temp
            os.remove(temp_path)