# Functionality concerning the bounding boxes
import cv2
import numpy as np
import os

from utils.Constants import IMG_WIDTH, IMG_HEIGHT
from math import sqrt

from utils.PedNamesAndHashes import convertHashToModelName


###############################################################################
######################## Parsing to VisDrone Categories #######################
###############################################################################

# I use VisDrone Categories as follows:
VIDSDRONE_OBJECT_CATEGORY_TO_NUMBER = {'pedestrian': 0,
                             'people': 1,
                             'bicycle': 2,
                             'car': 3,
                             'van': 4,
                             'truck': 5,
                             'tricycle': 6,
                             'awning-tricycle': 7,
                             'bus': 8,
                             'motor': 9}


with open(os.path.normpath("utils/vehicle_names_and_categories.csv"), "r") as namefile:
    namedata = namefile.read()
namedata = namedata.split("\n")
namedata = [n.split(",") for n in namedata]
namedata = {n[0].lower(): n[2] for n in namedata}
VEHICLE_NAME_TO_CATEGORY = namedata

VEHICLE_NAME_TO_CATEGORY['comet'] = 'Sports'
VEHICLE_NAME_TO_CATEGORY['astrope'] = 'Sedans'

# For those I am not sure of the specific category, but they are definitely cars, so setting them to 'Sports' is fine
VEHICLE_NAME_TO_CATEGORY['f'] = 'Sports'
VEHICLE_NAME_TO_CATEGORY['fq'] = 'Sports'
VEHICLE_NAME_TO_CATEGORY['schwarze'] = 'Sports'
VEHICLE_NAME_TO_CATEGORY['firetruk'] = 'Commercials'
VEHICLE_NAME_TO_CATEGORY['dominato'] = 'Sports'
VEHICLE_NAME_TO_CATEGORY['fork'] = 'UNRECOGNIZED_CATEGORY'
VEHICLE_NAME_TO_CATEGORY['issi'] = 'Compacts'
VEHICLE_NAME_TO_CATEGORY['ambulan'] = 'Commercials'
VEHICLE_NAME_TO_CATEGORY['carboniz'] = 'Sports'
VEHICLE_NAME_TO_CATEGORY['feltzer'] = 'Sports'
VEHICLE_NAME_TO_CATEGORY['buccanee'] = 'Sports'
VEHICLE_NAME_TO_CATEGORY['landstal'] = 'SUVs'
VEHICLE_NAME_TO_CATEGORY['furore'] = 'Sports'
VEHICLE_NAME_TO_CATEGORY['utiltruc'] = 'Commercials'
VEHICLE_NAME_TO_CATEGORY['bfinject'] = 'SUVs'
VEHICLE_NAME_TO_CATEGORY['rancherx'] = 'SUVs'
VEHICLE_NAME_TO_CATEGORY['roosevelt'] = 'Sports'
VEHICLE_NAME_TO_CATEGORY['cavcade'] = 'SUVs'
VEHICLE_NAME_TO_CATEGORY['washingt'] = 'Sports'
VEHICLE_NAME_TO_CATEGORY['rentbus'] = 'Sports'
VEHICLE_NAME_TO_CATEGORY['dilettan'] = 'Compacts'
VEHICLE_NAME_TO_CATEGORY['sandkin'] = 'SUVs'
VEHICLE_NAME_TO_CATEGORY['schafter'] = 'Sports'

# Custom added not found cars
VEHICLE_NAME_TO_CATEGORY['policeo'] = 'Sports'
VEHICLE_NAME_TO_CATEGORY['tailgate'] = 'Sports'


# This mapping is made according to 
# https://wiki.gtanet.work/index.php?title=Vehicle_Models
GTAV_CATEGORY_TO_VISDRONE_CATEGORY = {'Boats': 'UNRECOGNIZED_CATEGORY', # 'boat', # This should not happen
                                      'Commercials': 'truck',
                                      'Compacts': 'car',
                                      'Coupes': 'car',
                                      'Cycles': 'bicycle',
                                      'Emergency': 'MANUAL_CHECK',
                                      'Helicopters': 'UNRECOGNIZED_CATEGORY', # 'helicopter'
                                      'Industrial': 'MANUAL_CHECK',
                                      'Military': 'MANUAL_CHECK',
                                      'Motorcycles': 'motor',
                                      'Muscle': 'MANUAL_CHECK',
                                      'Off-Road': 'MANUAL_CHECK',
                                      'Planes': 'UNRECOGNIZED_CATEGORY',
                                      'SUVs': 'van',
                                      'Sedans': 'car',
                                      'Service': 'MANUAL_CHECK',
                                      'Sports': 'car',
                                      'Sports Classic': 'car',
                                      'Sports Classics': 'car',
                                      'Super': 'car',
                                      'Trailer': 'UNRECOGNIZED_CATEGORY',
                                      'Trains': 'UNRECOGNIZED_CATEGORY',
                                      'Utility': 'MANUAL_CHECK',
                                      'Vans': 'van',
                                      'Uncategorized': 'UNRECOGNIZED_CATEGORY',
                                      
                                      'UNRECOGNIZED_CATEGORY': 'UNRECOGNIZED_CATEGORY'}


MANUAL_CATEGORY = {#Emergency:
                   'Ambulance': 'truck',
                   'FBI': 'car',
                   'FBI2': 'van',
                   'FireTruck': 'truck',
                   'PBus': 'bus',
                   'Police': 'car',
                   'Police2': 'car',
                   'Police3': 'car',
                   'Police4': 'car',
                   'PoliceOld1': 'car',
                   'PoliceOld2': 'car',
                   'PoliceT': 'van',
                   'Policeb': 'motor',
                   'Polmav': 'UNRECOGNIZED_CATEGORY', # 'helicopter'
                   'Pranger': 'van',
                   'Predator': 'UNRECOGNIZED_CATEGORY', # 'boat'
                   'Riot': 'truck',
                   'Sheriff': 'car',
                   'Sheriff2': 'van',
                   #Industrial:
                   'Bulldozer': 'UNRECOGNIZED_CATEGORY', # other
                   'Cutter': 'UNRECOGNIZED_CATEGORY', #other
                   'Dump': 'truck',
                   'Flatbed': 'truck',
                   'Guardian': 'truck',
                   'Handler': 'UNRECOGNIZED_CATEGORY', # other
                   'Mixer': 'truck',
                   'Mixer2': 'truck',
                   'Rubble': 'truck',
                   'TipTruck': 'truck',
                   'TipTruck2': 'truck',
                    #Service:
                    'Airbus': 'bus',
                    'Brickade': 'truck',
                    'Bus': 'bus',
                    'Coach': 'bus',
                    'Rallytruck': 'truck',
                    'RentalBus': 'bus',
                    'Taxi': 'car',
                    'Tourbus': 'bus',
                    'Trash': 'truck',
                    'Trash2': 'truck',
                    #Utility:
                    'Airtug': 'UNRECOGNIZED_CATEGORY',
                    'Caddy': 'UNRECOGNIZED_CATEGORY',
                    'Caddy2': 'UNRECOGNIZED_CATEGORY',
                    'Caddy3': 'UNRECOGNIZED_CATEGORY',
                    'Docktug': 'UNRECOGNIZED_CATEGORY',
                    'Forklift': 'UNRECOGNIZED_CATEGORY',
                    'Mower': 'UNRECOGNIZED_CATEGORY',
                    'Ripley': 'truck',
                    'Sadler': 'van',
                    'Scrap': 'truck',
                    'TowTruck': 'truck',
                    'TowTruck2': 'truck',
                    'Tractor': 'UNRECOGNIZED_CATEGORY',
                    'Tractor2': 'UNRECOGNIZED_CATEGORY',
                    'Tractor3': 'UNRECOGNIZED_CATEGORY',
                    'TrailerLarge': 'UNRECOGNIZED_CATEGORY',
                    'TrailerS4': 'UNRECOGNIZED_CATEGORY',
                    'UtiliTruck': 'truck',
                    'UtiliTruck3': 'truck',
                    'UtiliTruck2': 'truck',
                    
                    # Extra
                    'rentbus': 'bus'}
MANUAL_CATEGORY = {k.lower(): v for k,v in MANUAL_CATEGORY.items()}

def getLabelFromObjectName(obj_name):
    # Make function case insensitive (all lists are lower case)
    obj_name = obj_name.lower()

    gtav_category = VEHICLE_NAME_TO_CATEGORY[obj_name]
    obj_category = GTAV_CATEGORY_TO_VISDRONE_CATEGORY[gtav_category]
    if obj_category == 'MANUAL_CHECK':
        if gtav_category == 'Military':
            MILITARY_NAMES = {'Rhino'}
            MILITARY_NAMES = {v.lower() for v in MILITARY_NAMES}
            if obj_name in MILITARY_NAMES:
                obj_category = 'UNRECOGNIZED_CATEGORY'
            else:
                obj_category = 'truck'
        elif gtav_category == 'Muscle':
            MUSCLE_NAMES = {'Moonbeam', 'Moonbeam2', 'RatLoader', 'RatLoader2', 'Sadler2', 'SlamVan', 'SlamVan2', 'SlamVan3'}
            MUSCLE_NAMES = {v.lower() for v in MUSCLE_NAMES}
            if obj_name in MUSCLE_NAMES:
                obj_category = 'van'
            else:
                obj_category = 'car'
        elif gtav_category == 'Off-Road':
            OFF_ROAD_NAMES = {'Bifta', 'Blazer', 'Blazer2', 'Blazer3', 'Blazer5', 'Dune4', 'Dune5'}
            OFF_ROAD_NAMES = {v.lower() for v in OFF_ROAD_NAMES}
            if obj_name in OFF_ROAD_NAMES:
                obj_category = 'UNRECOGNIZED_CATEGORY'
            else:
                obj_category = 'van'

        else:
            obj_category = MANUAL_CATEGORY[obj_name]
    
    return obj_category



# parse the bbboxes from label_aug format to VisDrone format. This also catches
# some wrong class labels by looking at the models, e.g. SUVs as  truck

# TODO We follow all conventions from VisDrone, while keeping the "tricycle" and
# "awning tricicly" class, despite of them not appearing in the game In
# accordance to VisDrone, people on bikes and motor are labeled as "people",
# while people in vehicles (e.g. in cars) are not labeled people that take part
# in the traffic (move) for example on sidewalks and crosswalks are labeled as
# "pedestrian", while people that don't take part (e.g. sit) are labeled as
# "people".
def parse_LabelAugToVisDrone(bboxes):
    items = bboxes.split("\n")
    ret = []
    for item in items[:-1]:
        data = item.split(" ")
        # Indices can be found in /DeepGTAV-PreSIL/dataformat-augmented.txt
        label = data[0]
        left = int(data[4])
        top = int(data[5])
        right = int(data[6])
        bottom = int(data[7])

        object_name = data[21]


        ignore_this_bbox = False


        # Convert Label to VisDrone Label

        try:
            if label == 'Airplane':
                ignore_this_bbox = True
            elif label == 'Animal':
                ignore_this_bbox = True
            elif label == 'Boat':
                ignore_this_bbox = True
            elif label == 'Bus':
                label = getLabelFromObjectName(object_name)        
            elif label == 'Car':
                label = getLabelFromObjectName(object_name)
            elif label == 'Cyclist':
                label = 'bicycle'
            elif label == 'Motorbike':
                label = getLabelFromObjectName(object_name)
            elif label == 'Pedestrian':
                label = 'pedestrian'
            elif label == 'Person_sitting':
                label = 'people'
            elif label == 'Railed':
                ignore_this_bbox = True
            elif label == 'Trailer':
                # Trailer Bounding Boxes on vehicles are wrong, but on standing trailers they are right
                # Also some of the standing Trailers are not labeled at all
                ignore_this_bbox = True
            elif label == 'Truck':
                label = getLabelFromObjectName(object_name)
            elif label == 'Utility':
                label = getLabelFromObjectName(object_name)
            # This does not seem to exist
            elif label == 'Van':
                label = getLabelFromObjectName(object_name)

            
            if label == 'UNRECOGNIZED_CATEGORY':
                # print('Encountered Unrecognized object_label in line: \n' + item)
                ignore_this_bbox = True
        except KeyError:

            # Not Found Labels are ignored. This should not be that important, but in the future it can be improved with the NotFoundObjectNames.txt
            label = object_name
            with open(os.path.normpath("utils/NotFoundObjectNames.txt"), "a") as not_found_file:
                not_found_file.write(object_name + "\n") # + "\n From Labels: \n" + bboxes)
            ignore_this_bbox = True

        if label == "pedestrian" or label=="people":
            # if the person is in a vehicle (data[22] != 0), find the vehicle
            # with the corresponding entity_id data[22]== vehData[15]
            # And append this person accordingly
            if data[22] != "0":
                thisVehicle="UNKNOWN"
                thisVehLabel = "UNKNOWN"
                thisVehModel = "UNKNOWN"
                for item2 in items[:-1]:
                    vehData = item2.split(" ")
                    if vehData[15] == data[22]:
                        thisVehLabel = vehData[0]
                        thisVehModel = vehData[21]
                        break

                if (thisVehLabel == "Motorbike" and getLabelFromObjectName(thisVehModel) == "motor") or thisVehLabel=="Cyclist":
                    label = "people"
                else:
                    ignore_this_bbox = True


        # ignore 1920 1080 0 0 boxes
        if not (left > right or top > bottom) and not ignore_this_bbox:
            ret.append({"label": label,"left": left,"top": top,"right": right,"bottom": bottom})
    return ret


def parseBBox2d_LikePreSIL(bbox2d):
    if bbox2d == None:
        return []
    items = bbox2d.split("\n")
    ret = []
    for item in items[:-1]:
        data = item.split(" ")
        # Indices can be found in /DeepGTAV-PreSIL/dataformat-augmented.txt
        label = data[0]
        left = int(data[4])
        top = int(data[5])
        right = int(data[6])
        bottom = int(data[7])

        # ignore 1920 1080 0 0 boxes
        if not (left > right or top > bottom):
            ret.append({"label": label,"left": left,"top": top,"right": right,"bottom": bottom})
    return ret


def parse_LabelAugToSeaDroneSea(bboxes):
    items = bboxes.split("\n")
    ret = []
    for item in items[:-1]:
        data = item.split(" ")
        # Indices can be found in /DeepGTAV-PreSIL/dataformat-augmented.txt
        label = data[0]
        left = int(data[4])
        top = int(data[5])
        right = int(data[6])
        bottom = int(data[7])

        object_name = data[21]
        ignore_this_bbox = True

        if label == 'Boat':
            label='boat'
            ignore_this_bbox=False
        elif label == 'Pedestrian':
            label = 'people'
            ignore_this_bbox=False
        elif label == 'Person_sitting':
            label = 'people'
            ignore_this_bbox=False

        if label=="people":
            # Check if it is in a vehicle (on a boat)
            # and check if it has a swimming west
            if data[22] == "0": # Not in a vehicle
                if convertHashToModelName(int(object_name)) == "s_m_y_baywatch_01":
                    label="peopleWithSwimwest"
                else:
                    label="people"
            else:
                if convertHashToModelName(int(object_name)) == "s_m_y_baywatch_01":
                    label="peopleOnBoatWithSwimwest"
                else:
                    label="peopleOnBoat"



        # ignore 1920 1080 0 0 boxes
        if not (left > right or top > bottom) and not ignore_this_bbox:
            ret.append({"label": label,"left": left,"top": top,"right": right,"bottom": bottom})
    return ret


def parse_LabelAugToCattle(bbox2d):
    if bbox2d == None:
        return []
    items = bbox2d.split("\n")
    ret = []
    for item in items[:-1]:
        data = item.split(" ")
        # Indices can be found in /DeepGTAV-PreSIL/dataformat-augmented.txt
        label = data[0].lower()
        left = int(data[4])
        top = int(data[5])
        right = int(data[6])
        bottom = int(data[7])

        object_name = data[21]

        if label=="Animal" or label=='animal':

            if convertHashToModelName(int(object_name)) == "a_c_cow":
                label="cow"

        ignore_this_bbox=False
        if label!="cow":
            ignore_this_bbox = True

        # ignore 1920 1080 0 0 boxes
        if not (left > right or top > bottom) and not ignore_this_bbox:
            ret.append({"label": label,"left": left,"top": top,"right": right,"bottom": bottom})
    return ret


###############################################################################
########################## Appending BBoxes to Images #########################
###############################################################################

def add_bboxes(image, bboxes, show_labels=True):
    """Add bounding boxes to an image

    bboxes as list of dicts, e.g.
        [{'category': 'Car',
          'left': '537',
          'top': '117',
          'right': '585',
          'bottom': '204'},
         {'category': 'Car',
          'left': '546',
          'top': '385',
          'right': '595',
          'bottom': '468'},
         {'category': 'Car',
          'left': '792',
          'top': '617',
          'right': '837',
          'bottom': '704'},
         {'category': 'Car',
          'left': '683',
          'top': '251',
          'right': '741',
          'bottom': '336'}]

    
    Args:
        image: the image to add bounding boxes into
        bboxes: the bounding box data
    """

    for bbox in bboxes:
        x1 = bbox['left']
        y1 = bbox['top']
        x2 = bbox['right']
        y2 = bbox['bottom']
        label = bbox['label']
        if show_labels:
            color = (255, 255, 0)
            # if x2 - x1 <= 10 or y2 - y1 <= 10:
            #     color = (0,255,255)
            cv2.putText(image, label, (x1, y1+25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, thickness = 2, lineType=cv2.LINE_AA) 
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    return image


def show_image_with_bboxes(image, bboxes): 
    # image = np.array(image)
    # image = image[...,::-1]
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image = add_bboxes(image, bboxes)

    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("test", 1920, 1080)
    cv2.imshow("test", image)
    cv2.waitKey(-1)



###############################################################################
########################## General Parsing Functions ##########################
###############################################################################

# convert a dictionary with format {"left", "right", "top", "bottom"} to list with format [x_center, y_center, width, height]
def convertDictLRTBToListXYWH(bboxes):
    bboxes_new =  [[b['label'], (b['left'] + b['right']) / 2, (b['top'] + b['bottom']) / 2, b['right'] - b['left'], b['bottom'] - b['top']] for b in bboxes]
    return bboxes_new

def convertListXYWHToDictLRTB(bboxes):
    bboxes_new = [{'label': b[0], 'left': int(b[1] - b[3]/2), 'right': int(b[1] + b[3]/2), 'top': int(b[2] - b[4]/2), 'bottom': int(b[2] + b[4]/2)} for b in bboxes]
    return bboxes_new

def convertLabelNamesToNumber(bboxes, labelToNumberDict):
    bboxes_new =  [[labelToNumberDict[b[0]], b[1], b[2], b[3], b[4]] for b in bboxes]
    return bboxes_new

def convertNumberToLabelNames(bboxes, labelToNumberDict):
    numberToLabelDict = {v: k for k,v in labelToNumberDict.items()}
    return [[numberToLabelDict[b[0]], b[1], b[2], b[3], b[4]] for b in bboxes]


# Note that there are some rounding Errors when converting from relative to
# absolute values and back
def convertBBoxes_AbsoluteToRelative(bboxes, img_width, img_height):
    bboxes_new = [[b[0], b[1] / img_width, b[2] / img_height, b[3] / img_width, b[4] / img_height] for b in bboxes]
    return bboxes_new

def convertBBoxes_RelativeToAboslute(bboxes, img_width, img_height):
    bboxes_new = [[b[0], int(b[1] * img_width), int(b[2] * img_height), int(b[3] * img_width), int(b[4] * img_height)] for b in bboxes]
    return bboxes_new



def revertConvertBBoxVisDroneToYolo(bboxes):
    bboxes = convertNumberToLabelNames(bboxes, VIDSDRONE_OBJECT_CATEGORY_TO_NUMBER)
    bboxes = convertListXYWHToDictLRTB(bboxes)
    return bboxes

def revertConvertBBoxVisDroneToYolo_ONLY_NUMBER(bboxes):
    bboxes = [[str(b[0]), b[1], b[2], b[3], b[4]] for b in bboxes]
    bboxes = convertListXYWHToDictLRTB(bboxes)
    # bboxes_new = [{'label': str(b[0]), 'left': int(b[1] - b[3]/2), 'right': int(b[1] + b[3]/2), 'top': int(b[2] - b[4]/2), 'bottom': int(b[2] + b[4]/2)} for b in bboxes]
    return bboxes


# parse string BBoxes in the format as it is used for YOLO training (space and
# newline separated (label, x_center, y_center, width, height) relative to the
# image size) to a list
def parseYoloBBoxStringToList(bboxes):
    if bboxes == "":
        return []
    bboxes = [bbox.split(" ") for bbox in bboxes.split("\n")]
    bboxes = [[int(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4])] for b in bboxes]
    return bboxes

def parseListToYoloBBoxString(bboxes):
    bboxes = ["{:d} {:1.6f} {:1.6f} {:1.6f} {:1.6f}".format(*bbox) for bbox in bboxes]
    bboxes = "\n".join(bboxes)
    return bboxes

def parseBBox_YoloFormatStringToImage(bboxes, img_width=IMG_WIDTH, img_height=IMG_HEIGHT):
    bboxes = parseYoloBBoxStringToList(bboxes)
    bboxes = convertBBoxes_RelativeToAboslute(bboxes, img_width, img_height)
    bboxes = revertConvertBBoxVisDroneToYolo(bboxes)
    return bboxes

def parseBBox_YoloFormatStringToImage_NumberOnly(bboxes, img_width=IMG_WIDTH, img_height=IMG_HEIGHT):
    bboxes = parseYoloBBoxStringToList(bboxes)
    bboxes = convertBBoxes_RelativeToAboslute(bboxes, img_width, img_height)
    bboxes = revertConvertBBoxVisDroneToYolo_ONLY_NUMBER(bboxes)
    return bboxes



# fully converts bounding boxes from label_aug to visdrone categories in format as required by yolo training
def parseBBoxesVisDroneStyle(bboxes, img_width=IMG_WIDTH, img_height=IMG_HEIGHT):
    bboxes = parse_LabelAugToVisDrone(bboxes)
    bboxes = convertDictLRTBToListXYWH(bboxes)
    bboxes = convertLabelNamesToNumber(bboxes, VIDSDRONE_OBJECT_CATEGORY_TO_NUMBER)
    bboxes = convertBBoxes_AbsoluteToRelative(bboxes, img_width, img_height)
    bboxes = parseListToYoloBBoxString(bboxes)
    return bboxes




# Parse the bbox2d string returned from DeepGTAV for the following conventions:
# People: 0
# People with Life Jackets: 1
# People On Boats: 2
# People On Boats with Life Jackets: 3
# Boats: 4
def parseBBoxesSeadroneSeaStyle(bboxes, img_width=IMG_WIDTH, img_height=IMG_HEIGHT):
    SEA_DRONE_SEA_OBJECT_CATEGORY_TO_NUMBER = {"people": 0, "peopleWithSwimwest": 1, "peopleOnBoat": 2, "peopleOnBoatWithSwimwest": 3, "boat":4}
    bboxes = parse_LabelAugToSeaDroneSea(bboxes)
    bboxes = convertDictLRTBToListXYWH(bboxes)
    bboxes = convertLabelNamesToNumber(bboxes, SEA_DRONE_SEA_OBJECT_CATEGORY_TO_NUMBER)
    bboxes = convertBBoxes_AbsoluteToRelative(bboxes, img_width, img_height)
    bboxes = parseListToYoloBBoxString(bboxes)
    return bboxes

def parseBBoxesCattleStyle(bboxes,  img_width=IMG_WIDTH, img_height=IMG_HEIGHT):
    CATTLE_OBJECT_CATEGORY_TO_NUMBER = {'cow': 0}
    bboxes = parse_LabelAugToCattle(bboxes)
    bboxes = convertDictLRTBToListXYWH(bboxes)
    bboxes = convertLabelNamesToNumber(bboxes, CATTLE_OBJECT_CATEGORY_TO_NUMBER)
    bboxes = convertBBoxes_AbsoluteToRelative(bboxes, img_width, img_height)
    bboxes = parseListToYoloBBoxString(bboxes)
    return bboxes
