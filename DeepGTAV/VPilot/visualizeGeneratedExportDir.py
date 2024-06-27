from PIL import Image
import cv2
import os
import numpy as np
import random
from utils.BoundingBoxes import parseBBox_YoloFormatStringToImage, show_image_with_bboxes, parseBBox_YoloFormatStringToImage_NumberOnly
    
def show_dataset_online_format(images_path, include_boats=False):
    images_path = os.path.normpath(images_path)
    for file_name in os.listdir(images_path):
        image = Image.open(os.path.join(images_path, file_name))
        with open(os.path.join(images_path.replace("images", "labels"), file_name.replace(".jpg", ".txt").replace(".png", ".txt")), "r") as bbox_file:
            bboxes = bbox_file.read()
        
        bboxes = parseBBox_YoloFormatStringToImage(bboxes, *image.size, include_boats=include_boats)
        
        image = np.array(image)
        show_image_with_bboxes(image, bboxes)

# show_dataset_online_format("Z:\\DeepGTAV-EXPORTDIR-TEST\\Generation3_With_Meta_data\\images")
# show_dataset_online_format("E:\\Bachelorarbeit\\Datasets\\VisDrone\\VisDrone_YOLO_FORMAT_CLEANED\\train\\images")
# show_dataset_online_format("G:\\EXPORTDIR\\ExportWater_4k_4\\images", include_boats=True)




def show_dataset_online_format_NUMBERS_ONLY(images_path):
    images_path = os.path.normpath(images_path)
    dir_ls = os.listdir(images_path)
    random.shuffle(dir_ls)
    for file_name in dir_ls:
        image = Image.open(os.path.join(images_path, file_name))
        with open(os.path.join(images_path.replace("images", "labels"), file_name.replace(".jpg", ".txt").replace(".png", ".txt")).replace(".JPG", ".txt"), "r") as bbox_file:
            bboxes = bbox_file.read()
        
        bboxes = parseBBox_YoloFormatStringToImage_NumberOnly(bboxes, *image.size)
        image = np.array(image)
        show_image_with_bboxes(image, bboxes)


# show_dataset_online_format_NUMBERS_ONLY("E:\\Bachelorarbeit\\Datasets\\VisDrone\\VisDrone_YOLO_FORMAT\\train\\images")
# show_dataset_online_format_NUMBERS_ONLY("E:\\Bachelorarbeit\\Datasets\\VisDrone\\VisDrone_YOLO_FORMAT_CLEANED\\train\\images")
# show_dataset_online_format_NUMBERS_ONLY("Z:\\visdrone-dataset\\VisDrone_YOLO_FORMAT\\train\\images")
# show_dataset_online_format_NUMBERS_ONLY("E:\\Bachelorarbeit\\Datasets\\VisDrone\\VisDrone_YOLO_FORMAT\\train\\images")
# show_dataset_online_format_NUMBERS_ONLY("Z:\\DeepGTAV-EXPORTDIR-TEST\\Generation1\\images")
# show_dataset_online_format_NUMBERS_ONLY("I:\\SeaDroneSee_1\\images")
# show_dataset_online_format_NUMBERS_ONLY("G:\\EXPORTDIR\\ExportWater_4k_6\\images")
# show_dataset_online_format_NUMBERS_ONLY("G:\\EXPORTDIR\\ExportStreet_1\\images")
# show_dataset_online_format_NUMBERS_ONLY("F:\\EXPORTDIR\\Testing_VisDrone_2\\images")
# show_dataset_online_format_NUMBERS_ONLY("E:\\Bachelorarbeit\\Datasets\\VisDrone\\VisDrone_YOLO_FORMAT\\train\\images")
# show_dataset_online_format_NUMBERS_ONLY("F:\\EXPORTDIR\\VisDrone_LowQuality_1\\images")
# show_dataset_online_format_NUMBERS_ONLY("F:\\EXPORTDIR\\VisDrone_1\\images")
# show_dataset_online_format_NUMBERS_ONLY("F:\\EXPORTDIR\\SeaDroneSee_1\\images")
# show_dataset_online_format_NUMBERS_ONLY("F:\\EXPORTDIR\\DGTA_SeaDroneSee\\val\\images")
# show_dataset_online_format_NUMBERS_ONLY("F:\\EXPORTDIR\\DGTA_VisDrone\\val\\images")
# show_dataset_online_format_NUMBERS_ONLY("F:\\EXPORTDIR\\DGTA_VisDrone_LowQuality\\train\\images")
# show_dataset_online_format_NUMBERS_ONLY("Z:\\DeepGTA\\Training\\SeaDronesSee\\train\\images")
# show_dataset_online_format_NUMBERS_ONLY("G:\\Exportdir\\TEST_DGTA_Cattle_1\\images")
# show_dataset_online_format_NUMBERS_ONLY("G:\\Exportdir\\TEST_DGTA_Cattle_2\\images")
# show_dataset_online_format_NUMBERS_ONLY("G:\\DGTA_Cattle_DAY\\val\\images")
show_dataset_online_format_NUMBERS_ONLY("Z:\\cattle\\CATTLE\\CattlePre\\images")


if __name__ == '__main__':
    pass