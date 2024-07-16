import os
import multiprocessing
from PIL import Image

def resize_image(image_path, output_path, size):
    with Image.open(image_path) as img:
        img = img.resize(size, Image.ANTIALIAS)
        img.save(output_path)

def process_images(input_dir, output_dir, size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    tasks = []

    for image_file in image_files:
        input_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, image_file)
        tasks.append((input_path, output_path, size))

    with multiprocessing.Pool() as pool:
        pool.starmap(resize_image, tasks)

if __name__ == "__main__":
    input_directory = 'path/to/your/input_directory'
    output_directory = 'path/to/your/output_directory'
    new_size = (800, 600)  # Example size

    process_images(input_directory, output_directory, new_size)
