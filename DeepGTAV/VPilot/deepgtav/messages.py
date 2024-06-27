#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
from numpy.lib.stride_tricks import as_strided

import utils.PedNamesAndHashes as PedNamesAndHashes
from utils.Constants import FRAME, SCREEN_RESOLUTION

class Scenario:
    def __init__(self, location=None, time=None, weather=None, vehicle=None, drivingMode=None, spawnedEntitiesDespawnSeconds=None):
        self.location = location #[x,y,z,heading] (heading optional)
        self.time = time #[hour, minute]
        self.weather = weather #string
        self.vehicle = vehicle #string
        self.drivingMode = drivingMode #[drivingMode, setSpeed]
        self.spawnedEntitiesDespawnSeconds = spawnedEntitiesDespawnSeconds # Despawn time in seconds


# TODO Default settings should be added in the future
class Dataset:
    def __init__(self, **kwargs):
            # rate=None, frame=None, screenResolution=None, vehicles=None, peds=None, trafficSigns=None, direction=None, reward=None, 
            # throttle=None, brake=None, steering=None, speed=None, yawRate=None, drivingMode=None, location=None, time=None,
            # offscreen=None, showBoxes=None, pointclouds=None, stationaryScene=None, vehiclesToCreate=None, pedsToCreate=None,
            # startIndex=None, lidarParam=None, collectTracking=None, recordScenario=None, positionScenario=None):
        self.__dict__.update(kwargs)
        
        if not 'frame' in kwargs:
            self.frame = FRAME
        if not 'screenResolution' in kwargs:
            self.screenResolution = SCREEN_RESOLUTION

        # self.rate = rate #Hz
        # self.frame = frame #[width, height]
        # self.screenResolution = screenResolution # [width, height]


        # self.vehicles = vehicles #boolean
        # self.peds = peds #boolean
        # self.trafficSigns = trafficSigns #boolean
        # self.direction = direction #[x,y,z]
        # self.reward = reward #[id, p1, p2]
        # self.throttle = throttle #boolean
        # self.brake = brake #boolean
        # self.steering = steering #boolean
        # self.speed = speed #boolean
        # self.yawRate = yawRate #boolean
        # self.drivingMode = drivingMode #boolean
        # self.location = location #boolean
        # self.time = time #boolean


        # self.offscreen = offscreen #boolean
        # self.showBoxes = showBoxes #boolean
        # self.pointclouds = pointclouds #boolean
        # self.stationaryScene = stationaryScene #boolean
        # self.vehiclesToCreate = vehiclesToCreate #array of [model, pos.forward, pos.right, heading, color]
        # self.pedsToCreate = pedsToCreate #array of peds
        # self.startIndex = startIndex #int
        # self.lidarParam = lidarParam #int
        # self.collectTracking = collectTracking #boolean
        # self.recordScenario = recordScenario #boolean (for recording clips)
        # self.positionScenario = positionScenario #boolean (for getting current location)

class Start:
    def __init__(self, scenario=None, dataset=None):
        self.scenario = scenario
        self.dataset = dataset

    def to_json(self):
        _scenario = None
        _dataset = None

        if (self.scenario != None):
            _scenario = self.scenario.__dict__

        if (self.dataset != None):
            _dataset = self.dataset.__dict__            

        return json.dumps({'start':{'scenario': _scenario, 'dataset': _dataset}})


class Config:
    def __init__(self, scenario=None, dataset=None):
        self.scenario = scenario
        self.dataset = dataset

    def to_json(self):
        _scenario = None
        _dataset = None

        if (self.scenario != None):
            _scenario = self.scenario.__dict__

        if (self.dataset != None):
            _dataset = self.dataset.__dict__            

        return json.dumps({'config':{'scenario': _scenario, 'dataset': _dataset}})

class Stop:
    def to_json(self):
        return json.dumps({'stop':None}) #super dummy

class Commands:
    def __init__(self, throttle=None, brake=None, steering=None):
        self.throttle = throttle #float (0,1)
        self.brake = brake #float (0,1)
        self.steering = steering #float (-1,1)

    def to_json(self):
        return json.dumps({'commands':self.__dict__})
        
def frame2numpy(frame, frameSize=SCREEN_RESOLUTION):
    buff = np.fromstring(frame, dtype='uint8')
    # Scanlines are aligned to 4 bytes in Windows bitmaps
    strideWidth = int((frameSize[0] * 3 + 3) / 4) * 4
    # Return a copy because custom strides are not supported by OpenCV.
    return as_strided(buff, strides=(strideWidth, 3, 1), shape=(frameSize[1], frameSize[0], 3)).copy()

# TODO not yet implemented in DeepGTAV-PreSIL
class StartRecording:
    def to_json(self):
        return json.dumps({'StartRecording':None}) #super dummy

# TODO not yet implemented in DeepGTAV-PreSIL
class StopRecording:
    def to_json(self):
        return json.dumps({'StopRecording':None}) #super dummy



# A command to use the AI in GTAV to walk to the specified location, from the current location
class GoToLocation:
    def __init__(self, x, y, z, speed = 20.0):
        self.x = x
        self.y = y
        self.z = z
        self.speed = speed
    
    def to_json(self):
        return json.dumps({'GoToLocation':self.__dict__})
    

# A command to Teleport to the specified location
class TeleportToLocation:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def to_json(self):
        return json.dumps({'TeleportToLocation':self.__dict__})


# A command to set the camera position and rotation (relative to the vehicle position and vehicle axis).
# Angles are in Degree 
# Default is to directly look forward

# in this context the x-axis is right, the y-axis is forward and the z-axis is up.
# Rotations are applied in the order rot_z, rot_y, rot_x
class SetCameraPositionAndRotation:
    def __init__(self, x=0, y=0, z=0, rot_x=0, rot_y=0, rot_z=0):
        self.x = x
        self.y = y 
        self.z = z
        self.rot_x = rot_x
        self.rot_y = rot_y
        self.rot_z = rot_z
    
    def to_json(self):
        return json.dumps({'SetCameraPositionAndRotation':self.__dict__})


# Creates a Pedestrian relative to the player vehicle. The model can be set with
# a model name, which can be found e.g. at:
# https://docs.fivem.net/docs/game-references/ped-models/ Note that there are
# different versions of most models, with different textures.

# Normally models wander in the game world. A specific animation can be given
# with an animDict and the animName. Those can be found at
# https://alexguirre.github.io/animations-list/
class CreatePed:
    def __init__(self, relativeForward = 0, relativeRight = 0, relativeUp = 0, model = None, heading = 0, task = 0, placeOnGround = True, animDict = "", animName = ""):
        
        if model == None:
            model = PedNamesAndHashes.getRandomPed()

        if isinstance(model, str):
            model = PedNamesAndHashes.convertModelNameToHash(model) 
        
        self.model = model
        self.relativeForward = relativeForward
        self.relativeRight = relativeRight
        self.relativeUp = relativeUp
        self.heading = heading
        self.task = task
        self.placeOnGround = placeOnGround
        self.animDict = animDict
        self.animName = animName
    
    def to_json(self):
        return json.dumps({'CreatePed':self.__dict__})


class CreateVehicle:
    def __init__(self, model="Blista", relativeForward=0, relativeRight=0, heading=0, color=0, color2=0, placeOnGround=True, withLifeJacketPed=False):
        self.model = model
        self.relativeForward = relativeForward
        self.relativeRight = relativeRight
        self.heading = heading
        self.color = color
        self.color2 = color2
        self.placeOnGround = placeOnGround
        self.withLifeJacketPed = withLifeJacketPed
    def to_json(self):
        return json.dumps({'CreateVehicle':self.__dict__})


# weather can be one of { "CLEAR", "EXTRASUNNY", "CLOUDS", "OVERCAST", "RAIN", "CLEARING", "THUNDER", "SMOG", "FOGGY", "XMAS", "SNOWLIGHT", "BLIZZARD", "NEUTRAL", "SNOW" }
class SetWeather:
    def __init__(self, weather):
        self.weather = weather
    
    def to_json(self):
        return json.dumps({'SetWeather':self.__dict__})

class SetClockTime:
    def __init__(self, hour = 0, minute = 0, second = 0):
        self.hour = hour 
        self.minute = minute
        self.second = second
    
    def to_json(self):
        return json.dumps({'SetClockTime':self.__dict__})
