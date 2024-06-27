#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils.Constants import IMG_WIDTH, IMG_HEIGHT


from deepgtav.messages import Start, Stop, Scenario, Dataset, Commands, frame2numpy, GoToLocation, TeleportToLocation, SetCameraPositionAndRotation
from deepgtav.messages import StartRecording, StopRecording, SetClockTime, SetWeather, CreatePed, CreateVehicle
from deepgtav.client import Client

from utils.BoundingBoxes import add_bboxes, parseBBox2d_LikePreSIL, parseBBoxesVisDroneStyle, parseBBox_YoloFormatStringToImage, parseYoloBBoxStringToList, parseListToYoloBBoxString, parseBBoxesCattleStyle
from utils.BoundingBoxes import parseBBoxesSeadroneSeaStyle
from utils.utils import save_image_and_bbox, save_meta_data, getRunCount, generateNewTargetLocation
from utils.colors import pickRandomColor

import argparse
import time
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from random import uniform
from math import sqrt
import numpy as np
import os

import base64
import open3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random


BOATS = ["Dinghy", "Dinghy2", "Dinghy3", "Dinghy4", "Jetmax", "Marquis", "Seashark", "Seashark2", "Seashark3", "Speeder", "Speeder2", "Squalo", # "Submersible", "Submersible2",
         "Suntrap", "Toro", "Toro2", "Tropic", "Tropic2", "Tug"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-l', '--host', default='127.0.0.1', help='The IP where DeepGTAV is running')
    parser.add_argument('-p', '--port', default=8000, help='The port where DeepGTAV is running')
    # parser.add_argument('-s', '--save_dir', default='C:\\Users\\kiefer\\Desktop\\DeepGTA\\DeepGTAV\\VPilot\\cattle_data', help='The directory the generated data is saved to')
    parser.add_argument('-s', '--save_dir', default='G:\EXPORTDIR\TEST_DGTA_Cattle_1', help='The directory the generated data is saved to')
    # args = parser.parse_args()

    # TODO for running in VSCode
    args = parser.parse_args('')
    
    args.save_dir = os.path.normpath(args.save_dir)

    client = Client(ip=args.host, port=args.port)

    scenario = Scenario(drivingMode=0, vehicle="buzzard", location=[245.23306274414062, -998.244140625, 29.205352783203125])
    dataset=Dataset(location=True, time=True, exportBBox2D=True)    
    
    client.sendMessage(Start(scenario=scenario, dataset=dataset))

    CAMERA_OFFSET_Z = -20
    CAMERA_ROT_X_LOW = -90
    CAMERA_ROT_X_HIGH = -20

    # Adjustments for recording from UAV perspective
    client.sendMessage(SetCameraPositionAndRotation(z = CAMERA_OFFSET_Z, rot_x = uniform(CAMERA_ROT_X_LOW, CAMERA_ROT_X_HIGH)))

    count = 0
    bbox2d_old = ""

    STARTING_COUNT = 100

    TRAVEL_HEIGHT_LOW = 30
    TRAVEL_HEIGHT_HIGH = 100

    currentTravelHeight = uniform(TRAVEL_HEIGHT_LOW, TRAVEL_HEIGHT_HIGH)

    xAr_min, xAr_max, yAr_min, yAr_max = 0, 1700, 1300, 2200
    # xAr_min, xAr_max, yAr_min, yAr_max = 500, 1000, 1500, 2000
    x_start, y_start = generateNewTargetLocation(xAr_min, xAr_max, yAr_min, yAr_max)
    x_target, y_target = generateNewTargetLocation(xAr_min, xAr_max, yAr_min, yAr_max)


    if not os.path.exists(os.path.join(args.save_dir, 'images')):
        os.makedirs(os.path.join(args.save_dir, 'images'))
    if not os.path.exists(os.path.join(args.save_dir, 'labels')):
        os.makedirs(os.path.join(args.save_dir, 'labels'))
    if not os.path.exists(os.path.join(args.save_dir, 'meta_data')):
        os.makedirs(os.path.join(args.save_dir, 'meta_data'))

    if not os.path.exists(os.path.join(args.save_dir, 'image')):
        os.makedirs(os.path.join(args.save_dir, 'image'))
    if not os.path.exists(os.path.join(args.save_dir, 'SegmentationAndBBox')):
        os.makedirs(os.path.join(args.save_dir, 'SegmentationAndBBox'))
    
    
    run_count = getRunCount(args.save_dir)
    weather = random.choice(["CLEAR", "EXTRASUNNY", "CLOUDS", "OVERCAST"])

    while True:
        try:
            count += 1
            print("count: ", count)

            # Only record every 10th frame
            if count > STARTING_COUNT and count % 10 == 0:
                client.sendMessage(StartRecording())
            if count > STARTING_COUNT and count % 10 == 1:
                client.sendMessage(StopRecording())

            # if count % 20 == 14:
            if count % 20 == 14:
                for i in [-60, -20, 20, 60]:
                    rand_x = uniform(-100,100)
                    rand_y = uniform(-15,15)

                    action = random.choice(["wander", "wander" "graze" "sit", "sit", "lie"])
                    if action == "wander":
                        client.sendMessage(CreatePed(150+rand_x, i+rand_y, heading=uniform(-180,180), model='a_c_cow'))
                    if action =="graze":
                        client.sendMessage(CreatePed(150+rand_x+10, i+rand_y+10, heading=uniform(-180,180), model='a_c_cow', animDict="creatures@cow@amb@world_cow_grazing@base", animName="base"))
                    if action == "sit":
                        client.sendMessage(CreatePed(150+rand_x+5, i+rand_y+5, heading=uniform(-180,180), model='a_c_cow', animDict="creatures@cow@move", animName="idle_sit"))
                    if action == "lie":
                        lie_left = random.choice([True, False])
                        if lie_left:
                            client.sendMessage(CreatePed(150+rand_x+15, i+rand_y+15, heading=uniform(-180,180), model='a_c_cow', animDict="creatures@cow@move", animName="dead_left"))
                        else:
                            client.sendMessage(CreatePed(150+rand_x+15, i+rand_y+15, heading=uniform(-180,180), model='a_c_cow', animDict="creatures@cow@move", animName="dead_right"))
                        
                    # dead_right

            # if count % 20 == 4:
            #     for i in [-60, -20, 20, 60]:
            #         rand_x = uniform(-100,100)
            #         rand_y = uniform(-15,15)
            #         withLifeJacketPed = random.choice([True, False])
            #         client.sendMessage(CreateVehicle(random.choice(BOATS), relativeForward=190+rand_x, relativeRight=i+rand_y, color=pickRandomColor(), color2=pickRandomColor(), heading=uniform(-180,180), withLifeJacketPed=withLifeJacketPed))


            if count == 2:
                client.sendMessage(TeleportToLocation(x_start, y_start, 200))
                client.sendMessage(GoToLocation(x_target, y_target, currentTravelHeight))
            if count == 4:
                client.sendMessage(SetClockTime(int(uniform(10,16))))

            # Make sure the weather is always "CLEAR"
            if count % 800 == 6:
                weather = random.choice(["CLEAR", "EXTRASUNNY", "CLOUDS", "OVERCAST"])
                client.sendMessage(SetWeather(weather))

            message = client.recvMessage()  

            # Generate a new Travelheight for every x frames (which are x / 10 recorded frames):
            if count % 500 == 0:
                currentTravelHeight = uniform(TRAVEL_HEIGHT_LOW, TRAVEL_HEIGHT_HIGH)
            
            if count % 700 == 0:
                client.sendMessage(SetCameraPositionAndRotation(z = CAMERA_OFFSET_Z, rot_x = uniform(CAMERA_ROT_X_LOW, CAMERA_ROT_X_HIGH)))

            estimated_ground_height = message["location"][2] - message["HeightAboveGround"]

            # Sometimes Generate a new location, to prevent getting stuck
            if count % 600 == 0:
                x_target, y_target = generateNewTargetLocation(xAr_min, xAr_max, yAr_min, yAr_max)
                client.sendMessage(GoToLocation(x_target, y_target, estimated_ground_height + currentTravelHeight))

            # if we are near the target location generate a new target location
            if sqrt((message["location"][0] - x_target) ** 2 + (message["location"][1] - y_target) ** 2) < 10:
                x_target, y_target = generateNewTargetLocation(xAr_min, xAr_max, yAr_min, yAr_max)
                client.sendMessage(GoToLocation(x_target, y_target, estimated_ground_height + currentTravelHeight))
                print("Going to new loctation: ", x_target, y_target, currentTravelHeight)


            # keep the currentTravelHeight below the wanted one
            # Move a little bit in the desired direction but primarily correct the height
            if message["HeightAboveGround"] > currentTravelHeight + 3 or message["HeightAboveGround"] < currentTravelHeight - 3:
                direction = np.array([x_target - message["location"][0], y_target - message["location"][1]])
                direction = direction / np.linalg.norm(direction)
                direction = direction * 50
                x_temporary = message["location"][0] + direction[0]
                y_temporary = message["location"][1] + direction[1]
                client.sendMessage(GoToLocation(x_temporary, y_temporary, estimated_ground_height + currentTravelHeight))
                print("Correcting height")
            else:
                client.sendMessage(GoToLocation(x_target, y_target, estimated_ground_height + currentTravelHeight))

            if message["bbox2d"] != bbox2d_old and message["bbox2d"] != None:
                filename = f'{run_count:04}' + '_' + f'{count:010}'
                
                bboxes =  parseBBoxesCattleStyle(message["bbox2d"])

                if bboxes != "":
                    save_image_and_bbox(args.save_dir, filename, frame2numpy(message['frame']), bboxes)
                    save_meta_data(args.save_dir, filename, message["location"], message["HeightAboveGround"], message["CameraPosition"], message["CameraAngle"], message["time"], weather)

                bbox2d_old = message["bbox2d"]

        except KeyboardInterrupt:
            break
            
    # We tell DeepGTAV to stop
    client.sendMessage(Stop())
    client.close()










