import requests
import os
from dotenv import load_dotenv

load_dotenv()

ACCESS_KEY = os.environ.get('UNSPLASH_ACCESS_KEY')
if not ACCESS_KEY:
    print('UNSPLASH_ACCESS_KEY environment variable not set')
    exit(1)

USER = 'kadykov'
DIR = 'data/images'
os.makedirs(DIR, exist_ok=True)

per_page = 10  # Lower to allow more pages within rate limit
total = 0
max_images = 100

# Get existing files to skip
existing_files = set(os.listdir(DIR))

for page in range(1, 11):  # Up to 10 pages
    url = f'https://api.unsplash.com/users/{USER}/photos?client_id={ACCESS_KEY}&page={page}&per_page={per_page}'
    response = requests.get(url)
    if response.status_code != 200:
        print(f'Error fetching page {page}: {response.status_code}')
        break
    photos = response.json()
    if not photos:
        break
    for photo in photos:
        if total >= max_images:
            break
        photo_id = photo['id']
        filename = f'{photo_id}.jpg'
        filepath = os.path.join(DIR, filename)
        if filename in existing_files:
            print(f'Skipping {filename}, already exists')
            continue
        img_url = photo['urls']['full']
        img_response = requests.get(img_url)
        if img_response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(img_response.content)
            total += 1
            existing_files.add(filename)
            print(f'Downloaded {filename}')
        else:
            print(f'Failed to download {filename}')
    if total >= max_images:
        break

print(f'Total images downloaded: {total}')