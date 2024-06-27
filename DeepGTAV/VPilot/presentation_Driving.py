#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils.Constants import IMG_WIDTH, IMG_HEIGHT


from deepgtav.messages import Start, Stop, Scenario, Dataset, Commands, frame2numpy, GoToLocation, TeleportToLocation, SetCameraPositionAndRotation
from deepgtav.messages import StartRecording, StopRecording, SetClockTime, SetWeather, CreatePed
from deepgtav.client import Client

from utils.BoundingBoxes import add_bboxes, parseBBox2d_LikePreSIL, parseBBoxesVisDroneStyle, parseBBox_YoloFormatStringToImage
from utils.utils import save_image_and_bbox, save_meta_data, getRunCount, generateNewTargetLocation

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
# import open3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-l', '--host', default='127.0.0.1', help='The IP where DeepGTAV is running')
    parser.add_argument('-p', '--port', default=8000, help='The port where DeepGTAV is running')
    parser.add_argument('-s', '--save_dir', default='F:\\EXPORTDIR\\Driving_presentation_1', help='The directory the generated data is saved to')
    # args = parser.parse_args()

    # TODO for running in VSCode
    args = parser.parse_args('')
    
    args.save_dir = os.path.normpath(args.save_dir)

    client = Client(ip=args.host, port=args.port)
    
    scenario = Scenario(drivingMode=[786603,15.0])
    dataset=Dataset(location=True, time=True, exportBBox2D=True, segmentationImage=True, exportLiDAR=True, maxLidarDist=120)    
    
    client.sendMessage(Start(scenario=scenario, dataset=dataset))

    # Camera offset in accordance with KITTY (in accordance with DeepGTA-PreSIL)
    client.sendMessage(SetCameraPositionAndRotation(z = 1.065))


    count = 0
    bbox2d_old = ""
    errors = []


    # SETTINGS

    x_start, y_start = -388, 0
    x_target, y_target = 1165, -553


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
    if not os.path.exists(os.path.join(args.save_dir, 'LiDAR')):
        os.makedirs(os.path.join(args.save_dir, 'LiDAR'))

    
    
        

    run_count = getRunCount(args.save_dir)


    messages = []
    emptybbox = []

    while True:
        try:
            count += 1
            print("count: ", count)

            # Only record every 10th frame
            if count > 50 and count % 10 == 0:
                client.sendMessage(StartRecording())
            if count > 50 and count % 10 == 1:
                client.sendMessage(StopRecording())
                

            # if count == 2:
            #     client.sendMessage(TeleportToLocation(-388, 0, 200))
            #     client.sendMessage(GoToLocation(1165, -553, 40))

            if count == 4:
                client.sendMessage(SetClockTime(12))

            if count == 250:
                client.sendMessage(SetClockTime(0))

            if count == 600:
                client.sendMessage(SetClockTime(19))
            

            message = client.recvMessage()  
            
            # None message from utf-8 decode error
            if message == None:
                continue

            # messages.append(message)

            # Plot Segmentation Image and Bounding Box image overlayed for testing 
            if message["segmentationImage"] != None and message["segmentationImage"] != "":
                bboxes = parseBBoxesVisDroneStyle(message["bbox2d"])
                
                filename = f'{run_count:04}' + '_' + f'{count:010}'
                save_image_and_bbox(args.save_dir, filename, frame2numpy(message['frame']), bboxes)
                save_meta_data(args.save_dir, filename, message["location"], message["HeightAboveGround"], message["CameraPosition"], message["CameraAngle"], message["time"], "CLEAR")
                
                bbox_image = add_bboxes(frame2numpy(message['frame'], (IMG_WIDTH,IMG_HEIGHT)), parseBBox_YoloFormatStringToImage(bboxes))
                
                nparr = np.fromstring(base64.b64decode(message["segmentationImage"]), np.uint8)
                segmentationImage = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)

                dst = cv2.addWeighted(bbox_image, 0.5, segmentationImage, 0.5, 0.0)

                cv2.namedWindow("SegmentationBBox", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("SegmentationBBox", int(1920 * (9/10)), int(1080 * (9/10)))
                cv2.imshow("SegmentationBBox", dst)
                cv2.waitKey(1)

                filename = f'{run_count:04}' + '_' + f'{count:010}' + ".png"
                cv2.imwrite(os.path.join(args.save_dir, "image", filename), bbox_image)
                cv2.imwrite(os.path.join(args.save_dir, "SegmentationAndBBox", filename), dst)


            if message["LiDAR"] != None and message["LiDAR"] != "":
                # print(message["LiDAR"])
                a = np.frombuffer(base64.b64decode(message["LiDAR"]), np.float32)
                # np.save(os.path.join(args.save_dir, "LiDAR", filename[:-4] + ".npy"), a)

                a = a.reshape((-1, 4))

                # Entities are stored in a[:, 3]
                # produce equally spaced entities for the colors
                unique_entities, inv = np.unique(a[:, 3], return_inverse=True)
                mapping = {k: v for k, v in zip(unique_entities, range(len(unique_entities)))}
                vals = np.array([mapping[key] for key in unique_entities])
                colors = np.array(vals[inv])

                points3d = np.delete(a, 3, 1)

                # point_cloud = open3d.geometry.PointCloud()
                # point_cloud.points = open3d.utility.Vector3dVector(points3d)
                # open3d.visualization.draw_geometries([point_cloud])

                fig = plt.figure(figsize=(80,40))
                ax = fig.add_subplot(111, projection='3d')
                ax.set_xlim([-5, 100])
                ax.set_ylim([-40, 40])
                ax.set_zlim([-2, 20])

                # ax.view_init(70, 180)
                ax.view_init(0, 180)

                ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([10, 1, 1, 1]))

                ax.scatter(points3d[:,0], points3d[:,1], points3d[:,2], c=colors, s=1)

                fig.canvas.draw()
                img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                height, width = fig.canvas.get_width_height()[::-1]
                img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                img = img[int(height * 0.5):int(height * 0.75), int(width * 0.4):int(width * 0.68)]
                cv2.namedWindow("LiDAR", cv2.WINDOW_NORMAL)
                cv2.imshow("LiDAR",img)
                cv2.resizeWindow("LiDAR", int(1920 * (9/10)), int(1080 * (9/10)))
                cv2.waitKey(1)

                cv2.imwrite(os.path.join(args.save_dir, "LiDAR", filename), img)

            
        except KeyboardInterrupt:
            break
            
    # We tell DeepGTAV to stop
    client.sendMessage(Stop())
    client.close()



