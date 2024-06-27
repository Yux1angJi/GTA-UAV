#Helper functions for exporting data
def getText(classID, distance, width, height, pixelCount):
    if (distance > 80 or #In metres
        height < 25 or
        width < 25 or
        pixelCount < 300):
        return "DontCare"
    return {
        0: "Car",
        1: "Motorcycle",
        2: "Cyclist",
        3: "QuadBike",
        7: "Train",
        10: "Pedestrian"
    }.get(classID,"DontCare")

def outputObjectInfo(text_file, instance, altBBox=True):
    #2D Bounding box
    bboxNums = instance['bbox2d']
    if altBBox:
        bboxNums = instance['bbox2dProcessed']

    #TODO Determine actual class of vehicle
    width = bboxNums[2] - bboxNums[0]
    height = bboxNums[3] - bboxNums[1]
    text = instance['objectType']#getText(instance['classID'], instance['distance'], width, height, instance['pointsHit2D'])
    text_file.write("%s" % text)

    #TODO - Determine if truncated
    text_file.write(" %f" % instance['truncation'])

    #TODO - Occlusion
    text_file.write(" %f" % instance['occlusion'])
    
    #Observation angle alpha TODO
    text_file.write(" %f" % instance['alpha'])

    for num in bboxNums:
        text_file.write(" %d" % num)
    
    #3D dimensions
    for num in instance['dimensions']:
        text_file.write(" %f" % num)
    
    #3D location
    for num in instance['location']:
        text_file.write(" %f" % num)
    
    #Rotation_y
    text_file.write(" %f" % instance['rotation_y'])
    
def printInstances(filename, list, augment, tracking=False, frameID=0, altBBox=True):
    text_file = open(filename, "a")
    for instance in list:
        if not altBBox or instance['pointsHit2D'] > 0:
            if tracking:
                text_file.write("%d" % frameID)
                text_file.write(" %d " % instance['entityID'])
                outputObjectInfo(text_file, instance, altBBox)
            else:
                outputObjectInfo(text_file, instance, altBBox)

                if augment:
                    #text_file.write(" Augmentations:")
                    text_file.write(" %d" % instance['entityID'])
                    text_file.write(" %d" % instance['pointsHit'])
                    text_file.write(" %f" % instance['speed'])
                    #text_file.write(" %f" % instance['heading'])
                    text_file.write(" %f" % instance['pitch'])
                    text_file.write(" %f" % instance['roll'])
                    text_file.write(" %s" % instance['modelString'])
                    
                    # if instance['offscreen']:
                    #     text_file.write(" 0")
                    # else:
                    #     text_file.write(" 1")

                    #3D dimensions offcenter
                    #for num in instance['offcenter']:
                    #    text_file.write(" %f" % num)
                
                    # for num in instance['FUR']:
                    #     text_file.write(" %f" % num)
                    # for num in instance['BLL']:
                    #     text_file.write(" %f" % num)

            text_file.write("\n")
    text_file.close()

def printCalib(filename, focalLen, width, height):
    text_file = open(filename, "w")
    for i in range(0,4):
        text_file.write("P%d:" % i)
        text_file.write(" %f 0 %d 0" % (focalLen, width/2))
        text_file.write(" 0 %f %d 0" % (focalLen, height/2))
        text_file.write(" 0 0 1 0\n")

    text_file.write("R0_rect: 1 0 0 0 1 0 0 0 1\n")
    #From KITTI velo to KITTI cam
    text_file.write("Tr_velo_to_cam: 0 -1 0 0 0 0 -1 0 1 0 0 0\n")
    text_file.write("Tr_imu_to_velo: 1 0 0 0 0 1 0 0 0 0 1 0")

    # From GTA to KITTI
    # text_file.write("Tr_velo_to_cam: 1 0 0 0 0 0 -1 0 0 1 0 0")
    text_file.close()