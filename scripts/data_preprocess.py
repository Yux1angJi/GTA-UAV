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


def rename_files(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for dir in dirs:
            if dir == 'meta_data':
                image_dir = os.path.join(root, dir)
                height = None
                
                # 获取高度信息
                parts = root.split(os.sep)
                for part in parts:
                    if part.startswith("H="):
                        height = part.split("=")[1]
                        break
                
                if height is None:
                    continue
                
                # 遍历 images 目录中的文件
                for image_file in os.listdir(image_dir):
                    if image_file.endswith(".txt"):
                        current_file_path = os.path.join(image_dir, image_file)
                        new_file_name = f"{height}_{image_file}"
                        new_file_path = os.path.join(image_dir, new_file_name)
                        
                        # 重命名文件
                        os.rename(current_file_path, new_file_path)
                        print(f"Renamed {current_file_path} to {new_file_path}")

if __name__ == "__main__":
    # input_directory = 'path/to/your/input_directory'
    # new_size = (512, 512)  # Example size

    # process_images(input_directory, new_size)
    base_dir = '/home/xmuairmud/data/GTA-UAV-data/randcam2_std0_stable_all/randcam2_std0_stable_all_resize'
    rename_files(base_dir)