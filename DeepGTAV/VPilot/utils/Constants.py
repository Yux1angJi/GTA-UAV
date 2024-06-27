# IMG_WIDTH = 1920
# IMG_HEIGHT = 1080


# We differ between 'frame' and 'screenResolution', where frame is the
# resolution of the depth/stencil buffer, while screenResolution is the
# resolution at which the game is finally displayed on the screen.

# I believe the following statements to be correct, but it is only
# tested for resolutions of 1920x1080, 3840x2160 and 3840x2160DSR. For
# resolutions smaller than 1920x1080 frame and screenResolution are the
# same. For larger resolutions the buffers have a resolution of 1/4 of
# the screen. E.g. when rendering the game with DSR at 16k (7680x4320),
# the buffers are at 4k (3840x2160) while the screenResolution is
# downsized to match the screen (in this case 4k).

MANAGED_SCREEN_RESOLUTION = "1920x1440"

if MANAGED_SCREEN_RESOLUTION != None:
    if MANAGED_SCREEN_RESOLUTION == "1920x1440":
        FRAME=[1920, 1440]
        SCREEN_RESOLUTION=[1920, 1440]
    elif MANAGED_SCREEN_RESOLUTION == "1920x1080":
        FRAME=[1920, 1080]
        SCREEN_RESOLUTION=[1920, 1080]
    elif MANAGED_SCREEN_RESOLUTION == "3840x2160":
        FRAME=[1920, 1080]
        SCREEN_RESOLUTION=[3840, 2160]
    elif MANAGED_SCREEN_RESOLUTION == "3840x2160DSR7680x4320":
        FRAME=[3840, 2160]
        SCREEN_RESOLUTION=[3840, 2160]
    else:
        raise ValueError("MANAGED_SCREEN_RESOLUTION not recognized, add the wanted settings to VPilot/utils/Constants.py")

IMG_WIDTH, IMG_HEIGHT = FRAME

