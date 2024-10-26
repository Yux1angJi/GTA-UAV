#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils.Constants import IMG_WIDTH, IMG_HEIGHT


from deepgtav.messages import *
from deepgtav.client import Client

from utils.BoundingBoxes import add_bboxes, parseBBox2d_LikePreSIL, parseBBoxesVisDroneStyle, parseBBox_YoloFormatStringToImage
from utils.utils import save_image_and_bbox, save_image, save_meta_data, getRunCount, generateNewTargetLocation
from utils.colors import pickRandomColor

import argparse
import time
import cv2

import base64

import matplotlib.pyplot as plt

from PIL import Image

from random import uniform
import random

from math import sqrt
import math
import numpy as np

import os
import sys



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


def calculate_projection_points(height, rot_x, rot_y, rot_z, temp_x, temp_y, hfov=60, vfov=38.9):

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-l', '--host', default='localhost', help='The IP where DeepGTAV is running')
    parser.add_argument('-p', '--port', default=8000, help='The port where DeepGTAV is running')
    parser.add_argument('-s', '--save_dir', default='D:\\data\\GTA-UAV\\Captured\\visual_test_lidar_120', help='The directory the generated data is saved to')
    args = parser.parse_args()

    client = Client(ip=args.host, port=args.port)
    # voltic
    # scenario = Scenario(drivingMode=786603, vehicle="buzzard", location=[245.23306274414062, -998.244140625, 29.205352783203125], spawnedEntitiesDespawnSeconds=200)
    scenario = Scenario(drivingMode=0, vehicle="buzzard", location=[245.23306274414062, -998.244140625, 29.205352783203125], spawnedEntitiesDespawnSeconds=200)
    dataset = Dataset(location=True, time=True, exportBBox2D=True, segmentationImage=True, exportLiDAR=True, maxLidarDist=12000)
    client.sendMessage(Start(scenario=scenario, dataset=dataset))
    message = client.recvMessage()
    
    # f = open('log.txt', 'w')
    # sys.stdout = f

    CAMERA_OFFSET_Z = -10
    CAMERA_OFFSET_ROT_Z = 20
    TRAVEL_HEIGHT = 60
    # TRAVEL_HEIGHT_LIST = [100, 200, 300, 400, 500, 600]
    TRAVEL_HEIGHT_ATEMPT = 100

    CAMERA_ROT_X = -90  # [-70, 110]
    CAMERA_ROT_X_L = -110
    CAMERA_ROT_X_R = -70

    CAMERA_ROT_Y = 0    # [-10, 10]
    CAMERA_ROT_Y_L = -10
    CAMERA_ROT_Y_R = 10

    CAMERA_ROT_Z = 0    # [-180, 180]
    CAMERA_ROT_Z_L = 300
    CAMERA_ROT_Z_R = 300

    STD_DEV = 5
    ERROR_EPS = 10

    rot_x = CAMERA_ROT_X
    rot_y = CAMERA_ROT_Y
    rot_z = CAMERA_ROT_Z + CAMERA_OFFSET_ROT_Z

    step = 100
    # STEP_LIST = [50, 100, 150, 200, 250, 300]
    STEP_LIST = [300]


    # Adjustments for recording
    #  from UAV perspective
    # client.sendMessage(SetCameraPositionAndRotation(z = CAMERA_OFFSET_Z, rot_x = uniform(CAMERA_ROT_X_LOW, CAMERA_ROT_X_HIGH)))
    # client.sendMessage(SetCameraPositionAndRotation(z = CAMERA_OFFSET_Z, rot_x=rot_x, rot_z=rot_z, rot_y=rot_y))
    client.sendMessage(SetCameraPositionAndRotation(z = -20, rot_x = -90))


    xAr_min, xAr_max, yAr_min, yAr_max = -3418, 3945, -3370, 7251

    x_step = step
    y_step = step
    x_y_list = [
        [-1431, -877],
        [-1376, -832],
        [-1268, -764],
        [-1140, -694],
        [-1045, -643],
        [-890, -572],
        [-672, -485],
        [-474, -411],
        [-290, -345],
        [-70, -265],
        [63, -215],
        [290, -138],
        [443, -87],
    ]

    z_loc = 0
    # x_target, y_target = generateNewTargetLocation(xAr_min, xAr_max, yAr_min, yAr_max)

    # run_count = getRunCount(save_dir)
    run_count = 1
    # weather = random.choice(["CLEAR", "EXTRASUNNY", "CLOUDS", "OVERCAST"])
    count = 0


    save_dir = f'{args.save_dir}'
    save_dir = os.path.normpath(save_dir)

    if not os.path.exists(os.path.join(save_dir, 'images')):
        os.makedirs(os.path.join(save_dir, 'images'))
    if not os.path.exists(os.path.join(save_dir, 'meta_data')):
        os.makedirs(os.path.join(save_dir, 'meta_data'))
    if not os.path.exists(os.path.join(save_dir, 'LiDAR')):
        os.makedirs(os.path.join(save_dir, 'LiDAR'))

    for i in range(len(x_y_list)):
        x_temp, y_temp = x_y_list[i][0], x_y_list[i][1]

        for f in range(60):
            count += 1

            if f == 1:
                weather = "CLEAR"
                client.sendMessage(SetWeather(weather))
            
            elif f == 2:
                client.sendMessage(SetClockTime(12))

            elif f == 3:
                client.sendMessage(TeleportToLocation(x_temp, y_temp, TRAVEL_HEIGHT_ATEMPT))
                message = client.recvMessage()

                heightAboveGround = message['HeightAboveGround']
                z_loc = message['location'][2]
                z_ground = z_loc - heightAboveGround
                z_loc = z_ground + TRAVEL_HEIGHT - CAMERA_OFFSET_Z
                z_temp = z_ground + TRAVEL_HEIGHT

                client.sendMessage(TeleportToLocation(x_temp, y_temp, z_loc))
            
            # elif f == 7:
            #     rot_x = gaussin_random_truncted(CAMERA_ROT_X_L, CAMERA_ROT_X_R, CAMERA_ROT_X, STD_DEV)
            #     rot_y = gaussin_random_truncted(CAMERA_ROT_Y_L, CAMERA_ROT_Y_R, CAMERA_ROT_Y, STD_DEV)
            #     rot_z = random.randint(CAMERA_ROT_Z_L, CAMERA_ROT_Z_R)
            #     # print(z_loc, heightAboveGround, TRAVEL_HEIGHT, z_temp)
            #     client.sendMessage(SetCameraPositionAndRotation(z = CAMERA_OFFSET_Z, rot_x=rot_x, rot_y=rot_y, rot_z=rot_z))
            #     message = client.recvMessage()
            #     rot_x_m, rot_y_m, rot_z_m = message['CameraAngle']
            #     pos_x_m, pos_y_m, pos_z_m = message['location']
            #     print('!!!!! rot', rot_x, rot_x_m, rot_y, rot_y_m, rot_z, rot_z_m)
            #     print('!!!!! pos', pos_x_m, pos_y_m, pos_z_m)
            elif f == 5:
                client.sendMessage(GoToLocation(443, -87, 40))

            elif f > 30 and f%10 == 0:
                client.sendMessage(StartRecording())

            elif f > 30 and f%10 == 1:
                client.sendMessage(StopRecording())
            
            message = client.recvMessage()

            if message["LiDAR"] != None and message["LiDAR"] != "":

                filename = f'{TRAVEL_HEIGHT}' + '_' + f'{run_count:04}' + '_' + f'{count:010}'

                x_temp, y_temp, z_temp = message['CameraPosition']
                heightAboveGround_2 = message['HeightAboveGround']

                rot_x, rot_y, rot_z = message['CameraAngle']

                proj_points = calculate_projection_points(heightAboveGround_2 + CAMERA_OFFSET_Z, rot_x, rot_y, rot_z, x_temp, y_temp)
                save_image(save_dir, filename, frame2numpy(message['frame']))
                save_meta_data(save_dir, filename, message["location"], message["HeightAboveGround"], proj_points, message["CameraPosition"], message["CameraAngle"], message["time"])

                # print(message["LiDAR"])
                a = np.frombuffer(base64.b64decode(message["LiDAR"]), np.float32)
                a = a.reshape((-1, 4))
                points3d = np.delete(a, 3, 1)

            #     # point_cloud = open3d.geometry.PointCloud()
            #     # point_cloud.points = open3d.utility.Vector3dVector(points3d)
            #     # open3d.visualization.draw_geometries([point_cloud])

                fig = plt.figure(figsize=(20,15))
                ax = fig.add_subplot(111, projection='3d')
                ax.view_init(30, - 90 - 90 -8)
                ax.scatter(points3d[:,0], points3d[:,1], points3d[:,2], c=points3d[:,2], s=2)
                plt.savefig(os.path.join(save_dir, "LiDAR", filename))

    # We tell DeepGTAV to stop
    client.sendMessage(Stop())
    client.close()

