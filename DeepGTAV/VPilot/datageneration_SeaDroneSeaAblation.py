#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils.Constants import IMG_WIDTH, IMG_HEIGHT


from deepgtav.messages import Start, Stop, Scenario, Dataset, Commands, frame2numpy, GoToLocation, TeleportToLocation, SetCameraPositionAndRotation
from deepgtav.messages import StartRecording, StopRecording, SetClockTime, SetWeather, CreatePed, CreateVehicle
from deepgtav.client import Client

from utils.BoundingBoxes import add_bboxes, parseBBox2d_LikePreSIL, parseBBoxesVisDroneStyle, parseBBox_YoloFormatStringToImage, parseYoloBBoxStringToList, parseListToYoloBBoxString
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
from datetime import datetime
import base64
import open3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random


BOATS = ["Dinghy", "Dinghy2", "Dinghy3", "Dinghy4", "Jetmax", "Marquis", "Seashark", "Seashark2", "Seashark3", "Speeder", "Speeder2", "Squalo", # "Submersible", "Submersible2",
         "Suntrap", "Toro", "Toro2", "Tropic", "Tropic2", "Tug"]

SEADRONESSEE_ALT_DISTRIBUTION = {11: 0.02957983193277311,
 9: 0.06084033613445378,
 20: 0.012773109243697478,
 40: 0.04134453781512605,
 57: 0.005378151260504202,
 59: 0.007394957983193277,
 61: 0.004033613445378151,
 63: 0.0003361344537815126,
 64: 0.0003361344537815126,
 65: 0.0003361344537815126,
 66: 0.0006722689075630252,
 67: 0.004369747899159664,
 68: 0.013445378151260505,
 70: 0.002689075630252101,
 71: 0.0003361344537815126,
 73: 0.0003361344537815126,
 75: 0.0003361344537815126,
 77: 0.0016806722689075631,
 79: 0.004705882352941176,
 81: 0.0003361344537815126,
 83: 0.0003361344537815126,
 85: 0.0003361344537815126,
 87: 0.0033613445378151263,
 89: 0.004033613445378151,
 91: 0.005714285714285714,
 92: 0.0003361344537815126,
 94: 0.0003361344537815126,
 96: 0.0003361344537815126,
 98: 0.013445378151260505,
 100: 0.0003361344537815126,
 102: 0.0003361344537815126,
 104: 0.0003361344537815126,
 106: 0.0020168067226890756,
 108: 0.002352941176470588,
 110: 0.0006722689075630252,
 112: 0.0003361344537815126,
 113: 0.0003361344537815126,
 115: 0.0006722689075630252,
 117: 0.0013445378151260505,
 119: 0.0033613445378151263,
 121: 0.0003361344537815126,
 10: 0.048403361344537814,
 43: 0.022521008403361343,
 48: 0.031932773109243695,
 37: 0.006386554621848739,
 38: 0.06084033613445378,
 50: 0.03831932773109244,
 30: 0.07092436974789916,
 56: 0.020168067226890758,
 58: 0.007058823529411765,
 60: 0.0010084033613445378,
 17: 0.004369747899159664,
 16: 0.0010084033613445378,
 15: 0.0006722689075630252,
 14: 0.002352941176470588,
 13: 0.0020168067226890756,
 12: 0.011764705882352941,
 28: 0.0003361344537815126,
 29: 0.0003361344537815126,
 31: 0.0003361344537815126,
 32: 0.0003361344537815126,
 33: 0.0003361344537815126,
 34: 0.0003361344537815126,
 36: 0.0003361344537815126,
 39: 0.0003361344537815126,
 41: 0.12033613445378151,
 230: 0.006386554621848739,
 238: 0.011764705882352941,
 239: 0.015798319327731094,
 130: 0.0006722689075630252,
 99: 0.006050420168067227,
 54: 0.013781512605042017,
 248: 0.015798319327731094,
 149: 0.005378151260504202,
 159: 0.0030252100840336134,
 166: 0.0036974789915966387,
 229: 0.011428571428571429,
 227: 0.005042016806722689,
 217: 0.004705882352941176,
 169: 0.012773109243697478,
 179: 0.004705882352941176,
 249: 0.008403361344537815,
 138: 0.005714285714285714,
 177: 0.0036974789915966387,
 118: 0.0030252100840336134,
 218: 0.005714285714285714,
 207: 0.0030252100840336134,
 219: 0.004369747899159664,
 126: 0.0010084033613445378,
 53: 0.005378151260504202,
 259: 0.004033613445378151,
 129: 0.002352941176470588,
 97: 0.0020168067226890756,
 199: 0.008067226890756302,
 246: 0.0010084033613445378,
 240: 0.0013445378151260505,
 170: 0.0020168067226890756,
 226: 0.002689075630252101,
 189: 0.0067226890756302525,
 78: 0.010084033613445379,
 128: 0.004033613445378151,
 55: 0.0033613445378151263,
 88: 0.006386554621848739,
 198: 0.004705882352941176,
 208: 0.002352941176470588,
 148: 0.004369747899159664,
 255: 0.0016806722689075631,
 109: 0.008739495798319327,
 188: 0.00773109243697479,
 69: 0.0016806722689075631,
 178: 0.005714285714285714,
 256: 0.0016806722689075631,
 74: 0.0006722689075630252,
 209: 0.004369747899159664,
 228: 0.0036974789915966387,
 168: 0.004705882352941176,
 197: 0.002689075630252101,
 116: 0.0003361344537815126,
 158: 0.0033613445378151263,
 186: 0.0010084033613445378,
 127: 0.0006722689075630252,
 204: 0.0003361344537815126,
 257: 0.0036974789915966387,
 139: 0.0033613445378151263,
 236: 0.0013445378151260505,
 155: 0.0006722689075630252,
 237: 0.0036974789915966387,
 258: 0.010084033613445379,
 196: 0.0003361344537815126,
 254: 0.0016806722689075631,
 157: 0.0003361344537815126,
 80: 0.0003361344537815126,
 51: 0.0003361344537815126,
 187: 0.002352941176470588,
 247: 0.0016806722689075631,
 206: 0.0006722689075630252,
 216: 0.0003361344537815126,
 120: 0.0006722689075630252,
 125: 0.0006722689075630252,
 124: 0.0003361344537815126,
 180: 0.0006722689075630252,
 76: 0.0003361344537815126,
 146: 0.0003361344537815126,
 167: 0.0003361344537815126,
 147: 0.0006722689075630252,
 190: 0.0010084033613445378,
 136: 0.0003361344537815126,
 205: 0.0003361344537815126,
 153: 0.0003361344537815126,
 52: 0.0003361344537815126,
 225: 0.0003361344537815126,
 156: 0.0006722689075630252,
 213: 0.0003361344537815126,
 72: 0.0003361344537815126,
 123: 0.0003361344537815126}

SEADRONESSEE_ANGLE_DISTRIBUTION = {45: 0.009747899159663866,
 24: 0.021848739495798318,
 23: 0.014453781512605042,
 22: 0.02722689075630252,
 21: 0.006386554621848739,
 20: 0.008403361344537815,
 19: 0.018823529411764704,
 18: 0.01546218487394958,
 17: 0.0067226890756302525,
 16: 0.006050420168067227,
 15: 0.0036974789915966387,
 35: 0.0036974789915966387,
 34: 0.026554621848739496,
 70: 0.004033613445378151,
 73: 0.0020168067226890756,
 74: 0.004369747899159664,
 75: 0.0010084033613445378,
 76: 0.0030252100840336134,
 -2: 0.0010084033613445378,
 2: 0.0003361344537815126,
 9: 0.0020168067226890756,
 12: 0.002689075630252101,
 13: 0.010756302521008404,
 14: 0.006386554621848739,
 25: 0.010420168067226891,
 27: 0.004033613445378151,
 8: 0.008067226890756302,
 10: 0.0020168067226890756,
 11: 0.0013445378151260505,
 58: 0.14016806722689076,
 54: 0.0003361344537815126,
 51: 0.0003361344537815126,
 50: 0.0030252100840336134,
 49: 0.0020168067226890756,
 47: 0.0020168067226890756,
 44: 0.010756302521008404,
 42: 0.008739495798319327,
 40: 0.008067226890756302,
 38: 0.008403361344537815,
 41: 0.0016806722689075631,
 46: 0.019495798319327733,
 43: 0.0013445378151260505,
 39: 0.01411764705882353,
 28: 0.005714285714285714,
 29: 0.0030252100840336134,
 31: 0.011428571428571429,
 33: 0.027899159663865546,
 36: 0.030252100840336135,
 52: 0.0006722689075630252,
 56: 0.0006722689075630252,
 63: 0.0003361344537815126,
 67: 0.0003361344537815126,
 71: 0.0003361344537815126,
 82: 0.0003361344537815126,
 86: 0.0003361344537815126,
 90: 0.39596638655462185,
 68: 0.02319327731092437,
 78: 0.0003361344537815126,
 62: 0.0003361344537815126,
 53: 0.020504201680672268,
 30: 0.002352941176470588,
 32: 0.0016806722689075631,
 26: 0.0036974789915966387,
 37: 0.0010084033613445378,
 48: 0.0016806722689075631,
 0: 0.0067226890756302525,
 5: 0.0003361344537815126,
 64: 0.0006722689075630252,
 72: 0.0006722689075630252,
 83: 0.006050420168067227,
 57: 0.0003361344537815126}

import json
# with open('altitude_angle_dist.txt') as f:
    # alt_ang = json.load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-l', '--host', default='127.0.0.1', help='The IP where DeepGTAV is running')
    parser.add_argument('-p', '--port', default=8000, help='The port where DeepGTAV is running')
    # parser.add_argument('-s', '--save_dir', default='D:\\DeepGTAV\\SeaDronesSee', help='The directory the generated data is saved to')
    parser.add_argument('-s', '--save_dir', default='F:\\DeepGTAV\\DGTA_SeaDroneSee_angle_aligned', help='The directory the generated data is saved to')
    args = parser.parse_args()

    # TODO for running in VSCode
    # args = parser.parse_args('')
    
    args.save_dir = os.path.normpath(args.save_dir)

    client = Client(ip=args.host, port=args.port)

    scenario = Scenario(drivingMode=0, vehicle="buzzard", location=[245.23306274414062, -998.244140625, 29.205352783203125])
    dataset=Dataset(location=True, time=True, exportBBox2D=True)    
    
    client.sendMessage(Start(scenario=scenario, dataset=dataset))

    CAMERA_OFFSET_Z = -20
    CAMERA_ROT_X_LOW = -90
    CAMERA_ROT_X_HIGH = -20

    # either one of altitude or angle or both can be aligned
    alt_aligned_to_seadronessee = False
    angle_aligned_to_seadronessee = True
    angle_and_alt = False
    assert not (angle_aligned_to_seadronessee and alt_aligned_to_seadronessee)


    # Adjustments for recording from UAV perspective
    if angle_aligned_to_seadronessee:
        # for seadronessee
        rot_x = np.random.choice(np.array(list(SEADRONESSEE_ANGLE_DISTRIBUTION.keys())),
                                 p=list(SEADRONESSEE_ANGLE_DISTRIBUTION.values()))
        rot_x = (-1)*float(rot_x)
        print('rot_x: ', rot_x)
        client.sendMessage(SetCameraPositionAndRotation(z=CAMERA_OFFSET_Z, rot_x=rot_x))
    else:
        client.sendMessage(SetCameraPositionAndRotation(z = CAMERA_OFFSET_Z, rot_x = uniform(CAMERA_ROT_X_LOW, CAMERA_ROT_X_HIGH)))

    count = 0
    bbox2d_old = ""

    STARTING_COUNT = 100


    if alt_aligned_to_seadronessee:
    #for seadronessee
        currentTravelHeight = np.random.choice(np.array(list(SEADRONESSEE_ALT_DISTRIBUTION.keys())),
                                           p=list(SEADRONESSEE_ALT_DISTRIBUTION.values()))
        currentTravelHeight = float(currentTravelHeight)
        currentTravelHeight = max(currentTravelHeight,30.0)
        print('currentTravelHeight: ', currentTravelHeight)
    else:
        #for uniform
        TRAVEL_HEIGHT_LOW = 20
        TRAVEL_HEIGHT_HIGH = 100
        currentTravelHeight = uniform(TRAVEL_HEIGHT_LOW, TRAVEL_HEIGHT_HIGH)
        print('currentTravelHeight: ', currentTravelHeight)

    xAr_min, xAr_max, yAr_min, yAr_max = -2800, -1800, -2500, -1300
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

            # if count / 10 > 100000:
            #     now = datetime.now()
            #     print(now)
            #     sys.exit()

            # Only record every 10th frame
            if count > STARTING_COUNT and count % 10 == 0:
                client.sendMessage(StartRecording())
            if count > STARTING_COUNT and count % 10 == 1:
                client.sendMessage(StopRecording())

            if count % 20 == 14:
                for i in [-60, -20, 20, 60]:
                    rand_x = uniform(-100,100)
                    rand_y = uniform(-15,15)
                    withLifeJacketPed = random.choice([True, False])
                    if withLifeJacketPed:
                        client.sendMessage(CreatePed(150+rand_x, i+rand_y, heading=uniform(-180,180), model='s_m_y_baywatch_01'))
                    else:
                        client.sendMessage(CreatePed(150+rand_x, i+rand_y, heading=uniform(-180,180), model=None))
            if count % 20 == 4:
                for i in [-60, -20, 20, 60]:
                    rand_x = uniform(-100,100)
                    rand_y = uniform(-15,15)
                    withLifeJacketPed = random.choice([True, False])
                    client.sendMessage(CreateVehicle(random.choice(BOATS), relativeForward=190+rand_x, relativeRight=i+rand_y, color=pickRandomColor(), color2=pickRandomColor(), heading=uniform(-180,180), withLifeJacketPed=withLifeJacketPed))


            if count == 2:
                client.sendMessage(TeleportToLocation(x_start, y_start, 200))
                client.sendMessage(GoToLocation(x_target, y_target, currentTravelHeight))
            if count % 400 == 0:
                # client.sendMessage(SetClockTime(int(uniform(0,24))))
                #adapted to be only during bright daylight
                client.sendMessage(SetClockTime(int(uniform(10,14))))

            # Make sure the weather is always "CLEAR"
            if count % 800 == 6:
                weather = random.choice(["CLEAR", "EXTRASUNNY", "CLOUDS", "OVERCAST"])
                client.sendMessage(SetWeather(weather))

            message = client.recvMessage()  

            # Generate a new Travelheight for every x frames (which are x / 10 recorded frames):
            if count % 500 == 0:
                if alt_aligned_to_seadronessee:
                    # for seadronessee
                    currentTravelHeight = np.random.choice(np.array(list(SEADRONESSEE_ALT_DISTRIBUTION.keys())),
                                                           p=list(SEADRONESSEE_ALT_DISTRIBUTION.values()))
                    currentTravelHeight = float(currentTravelHeight)
                    currentTravelHeight = max(currentTravelHeight,30.0)
                    print('currentTravelHeight: ', currentTravelHeight)
                elif angle_and_alt:
                    # for seadronessee
                    # currentTravelHeight = np.random.choice(np.array(list(SEADRONESSEE_ALT_DISTRIBUTION.keys())),
                    #                                        p=list(SEADRONESSEE_ALT_DISTRIBUTION.values()))
                    random_alt_ang = random.choice(alt_ang)
                    random_alt = float(random_alt_ang[0])
                    random_ang = float(random_alt_ang[1])
                    # currentTravelHeight = random_alt + 20
                    currentTravelHeight = max(currentTravelHeight,30.0)

                    print('currentTravelHeight: ', currentTravelHeight)
                else:
                    # for uniform
                    currentTravelHeight = uniform(TRAVEL_HEIGHT_LOW, TRAVEL_HEIGHT_HIGH)
                    print('currentTravelHeight: ', currentTravelHeight)
            
            if count % 500 == 0:
                if angle_aligned_to_seadronessee:
                    # for seadronessee
                    rot_x = np.random.choice(np.array(list(SEADRONESSEE_ANGLE_DISTRIBUTION.keys())),
                                                           p=list(SEADRONESSEE_ANGLE_DISTRIBUTION.values()))
                    rot_x = (-1)*float(rot_x)
                    print('rot_x: ', rot_x)
                    client.sendMessage(SetCameraPositionAndRotation(z = CAMERA_OFFSET_Z, rot_x = rot_x))
                elif angle_and_alt:
                    client.sendMessage(SetCameraPositionAndRotation(z=CAMERA_OFFSET_Z, rot_x=random_ang))
                else:
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

            # keep the currentTravelHeight under the wanted one
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
                
                bboxes =  parseBBoxesSeadroneSeaStyle(message["bbox2d"])

                if bboxes != "":
                    save_image_and_bbox(args.save_dir, filename, frame2numpy(message['frame']), bboxes)
                    save_meta_data(args.save_dir, filename, message["location"], message["HeightAboveGround"], message["CameraPosition"], message["CameraAngle"], message["time"], weather)

                bbox2d_old = message["bbox2d"]

        except KeyboardInterrupt:
            client.sendMessage(Stop())
            client.close()
            break
            
    # We tell DeepGTAV to stop
    client.sendMessage(Stop())
    client.close()










