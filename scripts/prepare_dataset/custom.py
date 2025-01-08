# ---------------------------------------------------------------
# Copyright (c) 2024-2025 Yuxiang Ji. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import os
import shutil
import cv2
import numpy as np
import albumentations as A
from torch.utils.data import Dataset
import copy
from tqdm import tqdm
import random
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import concurrent.futures
from PIL import Image
import math
from multiprocessing import Pool, cpu_count
import csv
import pickle
import random
import itertools
import json

Image.MAX_IMAGE_PIXELS = None



def copy_satellite():
    # Source directory containing the subdirectories
    source_dir = "/home/xmuairmud/data/vpair/tiles"

    # Destination directory to copy and rename the files
    destination_dir = "/home/xmuairmud/data/vpair/satellite"

    # Create the destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)

    # Iterate over each subdirectory in the source directory
    for sub_dir in os.listdir(source_dir):
        sub_dir_path = os.path.join(source_dir, sub_dir)
        
        # Check if it's a directory
        if os.path.isdir(sub_dir_path):
            for file_name in os.listdir(sub_dir_path):
                # Ensure it's a file
                file_path = os.path.join(sub_dir_path, file_name)
                if os.path.isfile(file_path):
                    # Construct new file name with the subdirectory name as prefix
                    new_file_name = f"{sub_dir}_{file_name}"
                    new_file_path = os.path.join(destination_dir, new_file_name)
                    
                    # Copy the file to the destination directory with the new name
                    shutil.copy(file_path, new_file_path)

    print("All files have been copied and renamed successfully.")


if __name__ == '__main__':
    copy_satellite()
