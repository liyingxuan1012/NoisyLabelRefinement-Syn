import os
import random
from PIL import Image


folder_path = 'ImageNet100/val/n01592084'
margin = 5

files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.JPEG'))]
random.shuffle(files)
files = files[:10]

num_columns = 5
num_rows = 2
single_image_width = 128
single_image_height = 128

total_width = num_columns * single_image_width + (num_columns - 1) * margin
total_height = num_rows * single_image_height + (num_rows - 1) * margin

output_image = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))

for index, file in enumerate(files):
    img = Image.open(file)
    img = img.resize((single_image_width, single_image_height))
    x = index % num_columns * (single_image_width + margin)
    y = index // num_columns * (single_image_height + margin)
    output_image.paste(img, (x, y))

output_image.save('samples.jpg')
