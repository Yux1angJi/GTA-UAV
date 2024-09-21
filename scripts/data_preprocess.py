import os
import multiprocessing
from PIL import Image
import math

def resize_image(image_path, save_path, size):
    with Image.open(image_path) as img:
        img = img.resize(size, Image.LANCZOS)
        img.save(save_path)

def process_images(input_dir, save_dir, size):
    tasks = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.png'):
                input_path = os.path.join(root, file)
                save_path = os.path.join(save_dir, file)
                tasks.append((input_path, save_path, size))

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


def area_of_rectangle(lat1, lon1, lat2, lon2):
    # 将经纬度从度转换为弧度
    lat1, lon1 = math.radians(lat1), math.radians(lon1)
    lat2, lon2 = math.radians(lat2), math.radians(lon2)
    
    # 地球半径 (单位：公里)
    R = 6371.0
    
    # 计算矩形区域的面积
    area = R**2 * abs(math.sin(lat2) - math.sin(lat1)) * abs(lon2 - lon1) * math.cos((lat1 + lat2) / 2)
    
    return abs(area)

if __name__ == "__main__":
    input_dir = '/home/xmuairmud/data/GTA-UAV-data/GTA-UAV-official/randcam2_std0_stable_5area/images'
    save_dir = '/home/xmuairmud/data/GTA-UAV-data/GTA-UAV-official/GTA-UAV-LR/drone/images'
    new_size = (512, 384)  # Example size

    process_images(input_dir, save_dir, new_size)

    # lat1 = 29.774065
    # lon1 = 115.970635
    # lat2 = 29.702283
    # lon2 = 115.996851

    # lat1 = 29.817376
    # lon1 = 116.033769
    # lat2 = 29.725402
    # lon2 = 116.064566

    # lat1 = 32.355491
    # lon1 = 119.805926
    # lat2 = 32.290290
    # lon2 = 119.900052

    # area = area_of_rectangle(lat1, lon1, lat2, lon2)
    # print(f"矩形区域的面积为: {area:.2f} 平方公里")