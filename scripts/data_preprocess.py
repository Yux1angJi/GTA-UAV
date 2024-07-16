import os
import multiprocessing
from PIL import Image

def resize_image(image_path, size):
    with Image.open(image_path) as img:
        img = img.resize(size, Image.ANTIALIAS)
        img.save(image_path)

def process_images(input_dir, size):
    tasks = []

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.png'):
                input_path = os.path.join(root, file)
                tasks.append((input_path, size))

    with multiprocessing.Pool() as pool:
        pool.starmap(resize_image, tasks)

if __name__ == "__main__":
    input_directory = 'path/to/your/input_directory'
    new_size = (512, 512)  # Example size

    process_images(input_directory, new_size)
