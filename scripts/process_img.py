from PIL import Image
import os
import shutil

    
def process():
    # 设置图像文件夹路径
    ori_dir = '/home/xmuairmud/data/visualization/drone'
    new_dir = '/home/xmuairmud/data/visualization/drone_resize'
    # 设置新的图像大小
    new_size = (960, 720)

    # 遍历目录中的所有文件
    for subdir, dirs, files in os.walk(ori_dir):
        for filename in files:
            if filename.endswith('.jpg') or filename.endswith('.png'):  # 检查文件格式
                image_path = os.path.join(subdir, filename)
                height = subdir.split('/')[-1]
                image_save_path = os.path.join(new_dir, f"{height}_{filename}")
                with Image.open(image_path) as img:
                    # 裁剪图像，删除顶部20行像素
                    left = 5
                    right = img.width - 5
                    top = 25
                    bottom = img.height - 5
                    img_cropped = img.crop((left, top, right, bottom))
                    # 调整图像大小
                    img_resized = img_cropped.resize(new_size)
                    # 保存修改后的图像，可以选择覆盖原图像或保存为新图像
                    img_resized.save(image_save_path)
                # print(f'Processed {filename}')


def copy_img():
    drone_dir = '/home/xmuairmud/data/visualization/drone_new'
    save_dir = '/home/xmuairmud/data/visualization/retrieval'

    retrieval_dir = '../Sample4Geo/visualization'

    img_list = os.listdir(drone_dir)

    for img_name in img_list:
        retrieval_img = f"cross_{img_name}"
        shutil.copy(os.path.join(retrieval_dir, retrieval_img), save_dir)




if __name__ == '__main__':
    process()
    # copy_img()