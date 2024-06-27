from PIL import Image
from random import uniform, sample
import cv2
import os






# saves image in np format and bboxes in string format
def save_image_and_bbox(save_dir, filename, image, bboxes):
    # convert image BGR -> RGB
    # image = image[...,::-1]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(image)
    img.save(os.path.join(save_dir, 'images', filename + ".jpg"))

    with open(os.path.join(save_dir, 'labels', filename + ".txt"), 'w') as file:
        file.write(bboxes)


# Format for the saved meta data is [x y z heightAboveGround, campos_x, campos_y, campos_z, camrot_x, camrot_y, camrot_z, time_hours, time_min, time_sec, weather]
def save_meta_data(save_dir, filename, location, heightAboveGround, cameraPosition, cameraRotation, time, weather):
    location = [str(i) for i in location]
    heightAboveGround = str(heightAboveGround)
    cameraPosition = [str(i) for i in cameraPosition]
    cameraRotation = [str(i) for i in cameraRotation]
    time = [str(i) for i in time]
    meta_text = " ".join(location) + " " + heightAboveGround + " " + " ".join(cameraPosition) + " " + " ".join(cameraRotation) + " " + " ".join(time) + " " + weather
    with open(os.path.join(save_dir, 'meta_data', filename + ".txt"), 'w') as file:
        file.write(meta_text)


# returns the highest run in the directory + 1
def getRunCount(save_dir):
    files = os.listdir(os.path.join(save_dir, 'images'))
    files = [int(f[:4]) for f in files]
    if files == []:
        return 0
    else:
        return max(files) + 1




# Go to some random location in area 
# x in [-1960, 1900]
# y in [-3360, 2000]
# This is the metropolitan area and some outskirts
# locations can be found here https://www.gtagmodding.com/maps/gta5/

def generateNewTargetLocation(x_min=-1960, x_max=1900, y_min=-3360, y_max=2000):
    x_target = uniform(x_min, x_max)
    y_target = uniform(y_min, y_max)
    return x_target, y_target

def getRandomWeather():
    weathers = {"CLEAR", "EXTRASUNNY", "CLOUDS", "OVERCAST", "RAIN", "CLEARING", "THUNDER", "SMOG", "FOGGY", "XMAS", "SNOWLIGHT", "BLIZZARD", "NEUTRAL", "SNOW"}
    return sample(weathers, 1)[0]