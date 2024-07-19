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

import matplotlib.pyplot as plt

from PIL import Image

from random import uniform
import random

from math import sqrt
import math
import numpy as np

import os

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


def calculate_projection_points(height, rot_x, rot_y, rot_z, temp_x, temp_y, hfov=60, vfov=36):

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
    print(relative_points)

    R = euler_to_rotation_matrix(pitch=rot_x, roll=rot_y, yaw=rot_z)

    actual_points = []
    for point in relative_points:
        rotated_point = R @ np.array(point)
        actual_x = temp_x + rotated_point[0]
        actual_y = temp_y + rotated_point[1]
        actual_points.append(actual_x)
        actual_points.append(actual_y)

    return actual_points


BICYCLES = ["Bmx", "Cruiser", "Fixter", "Scorcher", "TriBike", "TriBike2", "TriBike3"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-l', '--host', default='localhost', help='The IP where DeepGTAV is running')
    parser.add_argument('-p', '--port', default=8000, help='The port where DeepGTAV is running')
    parser.add_argument('-s', '--save_dir', default='F:\\GTA-UAV\\Captured', help='The directory the generated data is saved to')
    args = parser.parse_args()
    
    step = 1000
    save_dir = args.save_dir + f'\\{step}'

    args.save_dir = os.path.normpath(save_dir)

    client = Client(ip=args.host, port=args.port)
    
    scenario = Scenario(drivingMode=786603, vehicle="buzzard", location=[245.23306274414062, -998.244140625, 29.205352783203125], spawnedEntitiesDespawnSeconds=200)
    dataset=Dataset(location=True, time=True, exportBBox2D=True)

    client.sendMessage(Start(scenario=scenario, dataset=dataset))
    message = client.recvMessage()
    

    CAMERA_OFFSET_Z = -10
    CAMERA_ROT_X = -90

    rot_x = -90
    rot_y = 0
    rot_z = 0

    # Adjustments for recording
    #  from UAV perspective
    # client.sendMessage(SetCameraPositionAndRotation(z = CAMERA_OFFSET_Z, rot_x = uniform(CAMERA_ROT_X_LOW, CAMERA_ROT_X_HIGH)))
    client.sendMessage(SetCameraPositionAndRotation(z = CAMERA_OFFSET_Z, rot_x=-90, rot_z=0, rot_y=0))
    message = client.recvMessage()
    print('start camera', message['CameraAngle'])

    count = 0
    bbox2d_old = ""

    # SETTINGS
    STARTING_COUNT = 100


    OFFSET_HEIGHT = 30
    TRAVEL_HEIGHT = 200

    # currentTravelHeight = uniform(TRAVEL_HEIGHT_LOW, TRAVEL_HEIGHT_HIGH)
    
    ##### Whole Map
    xAr_min, xAr_max, yAr_min, yAr_max = -3418, 3945, -3370, 7251


    ### bottomLeft [-4000, -4000]
    ### topRight [8000, 6000]

    x_step = step
    y_step = step
    x_start, y_start = 1000, 1000
    z_loc = TRAVEL_HEIGHT
    # x_target, y_target = generateNewTargetLocation(xAr_min, xAr_max, yAr_min, yAr_max)


    if not os.path.exists(os.path.join(args.save_dir, 'images')):
        os.makedirs(os.path.join(args.save_dir, 'images'))
    if not os.path.exists(os.path.join(args.save_dir, 'labels')):
        os.makedirs(os.path.join(args.save_dir, 'labels'))
    if not os.path.exists(os.path.join(args.save_dir, 'meta_data')):
        os.makedirs(os.path.join(args.save_dir, 'meta_data'))
    

    run_count = getRunCount(args.save_dir)
    # weather = random.choice(["CLEAR", "EXTRASUNNY", "CLOUDS", "OVERCAST"])
    weather = "EXTRASUNNY"
    count = 0


    weather = "CLEAR"
    client.sendMessage(SetWeather(weather))

    client.sendMessage(SetClockTime(12))

    x_temp = 245.23306274414062
    y_temp = -998.244140625
    client.sendMessage(TeleportToLocation(x_temp, y_temp, TRAVEL_HEIGHT))
    message = client.recvMessage()
    # message = json.loads(data)
    heightAboveGround = message['HeightAboveGround']
    z_loc = message['CameraPosition'][2]

    print('height 1', heightAboveGround)
    print('camera loc 1', message['CameraPosition'])
    print('loc 1', message['location'])

    z_ground = z_loc- heightAboveGround
    z_loc = z_ground + TRAVEL_HEIGHT - CAMERA_OFFSET_Z

    client.sendMessage(TeleportToLocation(x_temp, y_temp, z_loc))
    message = client.recvMessage()

    x_temp, y_temp, z_temp = message['CameraPosition']
    heightAboveGround = message['HeightAboveGround']
    rot_x, rot_y, rot_z = message['CameraAngle']

    print('camera Rot', rot_x, rot_y, rot_z)
    print('height', heightAboveGround)

    points = calculate_projection_points(z_loc, rot_x, rot_y, rot_z, x_temp, y_temp)
    print('location:', message["location"])
    print('camera loc', x_temp, y_temp, z_temp)
    print('heightAboveGround', message['HeightAboveGround'])
    print('points', points)

    client.sendMessage(SetCameraPositionAndRotation(z = CAMERA_OFFSET_Z, rot_x=-90, rot_z=20, rot_y=0))
    message = client.recvMessage()
    print('end camera', message['CameraAngle'])

    a = input()

    # We tell DeepGTAV to stop
    client.sendMessage(Stop())
    client.close()

