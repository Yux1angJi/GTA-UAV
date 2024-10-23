#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils.Constants import IMG_WIDTH, IMG_HEIGHT

import base64
import open3d

from deepgtav.messages import *
from deepgtav.client import Client

from utils.BoundingBoxes import add_bboxes, parseBBox2d_LikePreSIL, parseBBoxesVisDroneStyle, parseBBox_YoloFormatStringToImage
from utils.utils import save_image_and_bbox, save_image, save_meta_data, getRunCount, generateNewTargetLocation
from utils.colors import pickRandomColor

import argparse
import time
import cv2

import matplotlib.pyplot as plt

from PIL import Image

from random import uniform
import random

from math import sqrt
import math
import numpy as np

import os
import sys

import gc



def gaussin_random_truncted(lower_bound, upper_bound, mean, std_dev):
    number = random.gauss(mean, std_dev)
    number = max(number, lower_bound)
    number = min(number, upper_bound)
    return number


def euler_to_rotation_matrix(pitch, roll, yaw):
    # Convert angles from degrees to radians
    pitch = np.radians(pitch)
    roll = np.radians(roll)
    yaw = np.radians(yaw)
    
    # Rotation matrix around x-axis (pitch)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])
    
    # Rotation matrix around y-axis (roll)
    Ry = np.array([
        [np.cos(roll), 0, np.sin(roll)],
        [0, 1, 0],
        [-np.sin(roll), 0, np.cos(roll)]
    ])
    
    # Rotation matrix around z-axis (yaw)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix
    R = Rz @ Ry @ Rx
    return R


def calculate_projection_points(height, rot_x, rot_y, rot_z, temp_x, temp_y, hfov=105, vfov=59):

    # rot_x, rot_y, rot_z represents pitch, roll, and yaw in eular system

    # Convert angles from degrees to radians
    hfov_rad = math.radians(hfov)
    vfov_rad = math.radians(vfov)
    rot_x = abs(rot_x + 90)
    tilt_angle_rad = math.radians(rot_x)

    # print(hfov_rad, vfov_rad, tilt_angle_rad)
    
    # Calculate the width and length of the projection on the ground
    W = 2 * height * math.tan(hfov_rad / 2)
    L = 2 * height * math.tan(vfov_rad / 2)
    
    # Calculate the shift in the projection center due to the tilt angle
    D = height * math.tan(tilt_angle_rad)
    
    # Calculate the four corner points
    P1 = (-W / 2, -L / 2 + D)
    P2 = (W / 2, -L / 2 + D)
    P3 = (-W / 2, L / 2 + D)
    P4 = (W / 2, L / 2 + D)
    relative_points = [
        [-W / 2, -L / 2 + D, 0],
        [W / 2, -L / 2 + D, 0],
        [-W / 2, L / 2 + D, 0],
        [W / 2, L / 2 + D, 0]
    ]
    # print(relative_points)

    R = euler_to_rotation_matrix(pitch=rot_x, roll=rot_y, yaw=rot_z)

    actual_points = []
    for point in relative_points:
        rotated_point = R @ np.array(point)
        actual_x = temp_x + rotated_point[0]
        actual_y = temp_y + rotated_point[1]
        actual_points.append(actual_x)
        actual_points.append(actual_y)

    return actual_points


def get_meta_data_from_file(meta_data_dir):
    meta_data_list = []

    for filename in os.listdir(meta_data_dir):
        with open(os.path.join(meta_data_dir, filename), 'r') as file:
            line = file.readline().strip()
            values = line.split()

            x = float(values[0])
            y = float(values[1])
            z = float(values[2])
   
            cam_roll = float(values[7])
            cam_pitch = float(values[8])
            cam_yaw = float(values[9])

            meta_data_list.append([filename.replace('.txt', ''), x, y, z, cam_roll, cam_pitch, cam_yaw])

    return meta_data_list



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-l', '--host', default='localhost', help='The IP where DeepGTAV is running')
    parser.add_argument('-p', '--port', default=8000, help='The port where DeepGTAV is running')
    parser.add_argument('-s', '--save_dir', default='D:\\data\\GTA-UAV\\Captured\\lidar_from_file', help='The directory the generated data is saved to')
    args = parser.parse_args()

    client = Client(ip=args.host, port=args.port)

    save_dir = f'{args.save_dir}'
    save_dir = os.path.normpath(save_dir)
    if not os.path.exists(os.path.join(save_dir, 'images')):
            os.makedirs(os.path.join(save_dir, 'images'))
    if not os.path.exists(os.path.join(save_dir, 'lidars')):
        os.makedirs(os.path.join(save_dir, 'lidars'))
    if not os.path.exists(os.path.join(save_dir, 'meta_data')):
        os.makedirs(os.path.join(save_dir, 'meta_data'))

    # voltic
    # scenario = Scenario(drivingMode=786603, vehicle="buzzard", location=[245.23306274414062, -998.244140625, 29.205352783203125], spawnedEntitiesDespawnSeconds=200)
    # scenario = Scenario(drivingMode=[786603,0], vehicle="buzzard", location=[245.23306274414062, -998.244140625, 29.205352783203125], spawnedEntitiesDespawnSeconds=200)
    scenario = Scenario(drivingMode=[786603,0], vehicle="voltic", location=[245.23306274414062, -998.244140625, 29.205352783203125], spawnedEntitiesDespawnSeconds=200)

    dataset = Dataset(location=True, time=True, exportLiDAR=True, maxLidarDist=5000)
    client.sendMessage(Start(scenario=scenario, dataset=dataset))
    
    # f = open('log.txt', 'w')
    # sys.stdout = f

    CAMERA_OFFSET_Z = -10
    CAMERA_OFFSET_ROT_Z = 20


    CAMERA_ROT_X = -90  # [-70, 110]
    CAMERA_ROT_X_L = -110
    CAMERA_ROT_X_R = -70

    CAMERA_ROT_Y = 0    # [-10, 10]
    CAMERA_ROT_Y_L = -10
    CAMERA_ROT_Y_R = 10

    CAMERA_ROT_Z = 0    # [-180, 180]
    CAMERA_ROT_Z_L = -180
    CAMERA_ROT_Z_R = 180


    client.sendMessage(SetCameraPositionAndRotation(z = CAMERA_OFFSET_Z, rot_x=CAMERA_ROT_X, rot_z=CAMERA_ROT_Z, rot_y=CAMERA_ROT_Y))

    count_start = 0

    meta_data_dir = 'D:\\data\\GTA-UAV\\Captured\\randcam2_std0_stable_5area\\meta_data'

    meta_data_list = get_meta_data_from_file(meta_data_dir)
    total_count = len(meta_data_list)

    for count in range(count_start, total_count):

        meta_data = meta_data_list[count]
        filename, x, y, z, cam_x, cam_y, cam_z = meta_data


        for f in range(10):
            if f == 1:
                weather = "CLEAR"
                client.sendMessage(SetWeather(weather))
                message = client.recvMessage()
            
            elif f == 2:
                client.sendMessage(SetClockTime(12))
                message = client.recvMessage()

            elif f == 3:
                client.sendMessage(TeleportToLocation(x, y, z))
                message = client.recvMessage()
                z_loc = message['location'][2]
                
            elif f == 4:
                if abs(z_loc - z) > 10:
                    print(f'Warning!! Z_loc and z diff is {abs(z_loc - z)}. Retry to teleport.')
                    client.sendMessage(TeleportToLocation(x_temp, y_temp, z))
                message = client.recvMessage()
            
            elif f == 6:
                # print(z_loc, heightAboveGround, TRAVEL_HEIGHT, z_temp)
                client.sendMessage(SetCameraPositionAndRotation(z = CAMERA_OFFSET_Z, rot_x=cam_x, rot_y=cam_y, rot_z=cam_z))
                message = client.recvMessage()
                rot_x_m, rot_y_m, rot_z_m = message['CameraAngle']
                pos_x_m, pos_y_m, pos_z_m = message['location']
                # print(f'Metadata rotx={cam_x}, roty={cam_y}, rotz={cam_z}, posx={x}, posy={y}, posz={z}')
                # print(f'Current rotx={rot_x_m}, roty={rot_y_m}, rotz={rot_z_m}, posx={pos_x_m}, posy={pos_y_m}, posz={pos_z_m}')

            elif f == 8:
                client.sendMessage(StartRecording())
                message = client.recvMessage()

                if message["LiDAR"] != None and message["LiDAR"] != "":

                    x_temp, y_temp, z_temp = message['CameraPosition']
                    heightAboveGround = message['HeightAboveGround']

                    rot_x, rot_y, rot_z = message['CameraAngle']
                    if rot_x > CAMERA_ROT_X_R or rot_x < CAMERA_ROT_X_L \
                        or rot_y > CAMERA_ROT_Y_R or rot_y < CAMERA_ROT_Y_L \
                            or rot_z > CAMERA_ROT_Z_R or rot_z < CAMERA_ROT_Z_L:
                        print(f'Warning!! camera rot error! rot_x={rot_x}, rot_y={rot_y}, rot_z={rot_z}')
                        continue
                    
                    proj_points = calculate_projection_points(heightAboveGround, rot_x, rot_y, rot_z, x_temp, y_temp)

                    lidar = np.frombuffer(base64.b64decode(message["LiDAR"]), np.float32)
                    lidar = lidar.reshape((-1, 4))
                    points3d = np.delete(lidar, 3, 1)

                    # 获取 Z 坐标并对其进行归一化
                    z_min, z_max = points3d[:, 0].min(), points3d[:, 0].max()
                    z_norm = (points3d[:, 0] - z_min) / (z_max - z_min)

                    # 创建 RGB 颜色数组，基于 Z 坐标的归一化值创建颜色梯度（例如，蓝色到红色）
                    colors = np.zeros((points3d.shape[0], 3))
                    colors[:, 0] = z_norm  # 红色分量随高度增加
                    colors[:, 2] = 1 - z_norm  # 蓝色分量随高度减少

                    point_cloud = open3d.geometry.PointCloud()
                    point_cloud.points = open3d.utility.Vector3dVector(points3d)
                    point_cloud.colors = open3d.utility.Vector3dVector(colors)
                    # open3d.visualization.draw_geometries([point_cloud])

                    open3d.io.write_point_cloud(os.path.join(args.save_dir, "lidars", filename + '.ply'), point_cloud)

                    save_image(save_dir, filename, frame2numpy(message['frame']))
                    save_meta_data(save_dir, filename, message["location"], message["HeightAboveGround"], proj_points, message["CameraPosition"], message["CameraAngle"], message["time"])

                    del point_cloud, lidar, points3d, colors, z_norm
                    del message
                    gc.collect()

                    print(f'{filename} Done, count={count}')
                    count += 1
            
                else:
                    print(f'Warning! LiDAR None at filename={filename}, count={count}, height={heightAboveGround}, rot_x={rot_x}, rot_y={rot_y}, rot_z={rot_z}')
                    continue

            elif f == 9:
                client.sendMessage(StopRecording())
                message = client.recvMessage()



            # if message["StencilImage"]!=None and message["StencilImage"]!="":
            #     print("stencilImage")
            # if message["DepthBuffer"]!=None and message["DepthBuffer"]!="":
            #     print("DepthBuffer")
            # if message["LiDARRaycast"]!=None and message["LiDARRaycast"]!="":
            #     print("LiDARRaycast")
            # if message["LiDAR"]!=None and message["LiDAR"]!="":
            #     print("LiDAR")


            else:
                message = client.recvMessage()
            
    # We tell DeepGTAV to stop
    client.sendMessage(Stop())
    client.close()

