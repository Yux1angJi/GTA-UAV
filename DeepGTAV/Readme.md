 	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/leveraging-synthetic-data-in-object-detection/object-detection-on-seadronessee)](https://paperswithcode.com/sota/object-detection-on-seadronessee?p=leveraging-synthetic-data-in-object-detection)

# DeepGTA
## Ground Truth Data Generation for UAV Object detection

The DeepGTAV framework can be used to produce ground truth training data with
high accuracy for different tasks, by leveraging the high visual fidelity
environment of GTAV.

In particular HD images, LiDAR, depth, stencil, labels, object segmentation and
bounding boxes can be exported for self-driving car and aerial scenarios.
Furthermore the system implements easy control through messages from a python
client, that make it suitable for custom scenarios and in principle even
reinforcement learning tasks. 


We used synthetic training data generated with the DeepGTAV framework to good
real world object detection performance reported in our paper `Leveraging
Synthetic Data in Object Detection on Unmanned Aerial Vehicles`
https://arxiv.org/abs/2112.12252.

## Video Demonstration
[![DeepGTAV Demo Video](/images/YoutubeThumbnail.jpg)](https://www.youtube.com/watch?v=h3By_ZOdlAc "DeepGTAV: Synthetic ML Training Data from GTAV")


Below we show some example images from DeepGTAV with the object bounding boxes
(right) and corresponding images from real world datasets (left)

![Example images from DeepGTAV with corresponding images from real world
datasets](/images/ExampleImages.png)

## Pregenerated Datasets
The datasets we used in work `Leveraging
Synthetic Data in Object Detection on Unmanned Aerial Vehicles`
(https://arxiv.org/abs/2112.12252) are available under
https://cloud.cs.uni-tuebingen.de/index.php/s/cQ3Qt5z8o4e5GWo


### Format of the Labels and Metadata in the Pregenerated Datasets:

The labels have the following id to category name mapping:

{"swimmer": 0, "floater": 1, "swimmer on boat": 2, "floater on boat": 3, "boat":4}

In each metadata file we have the following values separated by spaces:

    [x y z heightAboveGround, campos_x, campos_y, campos_z, camrot_x, camrot_y, camrot_z, time_hours, time_min, time_sec, weather]

- `x,y,z`: the location of the player vehicle in GTAV world coordinates.
- `heightAboveGround`: The height difference of the player vehicle to the ground (or water)
- `campos_x, campos_y, campos_z`: The positions of the camera in world coordinates.
- `camrot_x, camrot_y, camrot_z`: The camera rotations
- `time_hours, time_min, time_sec`: The ingame time
- `weather`: The ingame weather condition


Note that the camera position in general is offset wrt. the player vehicle. Also the height of the camera above ground is different from the heightAboveGround for the vehicle, so if you need this you need to calculate it. 


# Simple Use
In the following the steps that are necessary to just export data using DeepGTAV
are described. In principle DeepGTAV should work with any version of GTAV, if
the correct newest version of ScriptHookV is obtained. The last working version
for me employed GTAV from the Epic store in version 1.0.2060.1

## Installation of DeepGTAV
1. Install MS Visual Studio 2017
   (https://visualstudio.microsoft.com/de/vs/older-downloads/) with `Desktop
   developement with C++` and `Game Developement with C++` selected. This
   installs some dependencies, that are needed, otherwise DeepGTAV crashes
   without an error. 
2. Obtain the newest version of ScriptHookV from
   (http://dev-c.com/GTAV/scripthookv) and copy the files from `bin/` to
   `DeepGTAV-PreSIL/Release/` overwriting the old files (this is not necessery
   when using GTAV in version 1.0.2245.0 or older)
3. Copy the contents of `DeepGTAV-PreSIL/bin/Release/` to the GTAV install
   directory
4. Replace the save game data in `Documents/Rockstar Games/GTA V/Profiles/` with
   the contents of `DeepGTAV-PreSIL/bin/SaveGame`
5. Done!
6. Only if you are using the reinforcement learning rewarder: Download paths.xml
   (https://drive.google.com/file/d/0B6pR5O2YrmHnNU9EMDBSSFpMV00/view?usp=sharing)
   and store it in the GTAV install directory

## Game Settings
1. Set the game in windowed mode.
2. In the graphics setting MSAA has to be disabled, otherwise no objects are
   detected. The correct settings should be loaded by replacing the files in
   `Documents/Rockstar Games/GTA V/Profiles/`, but keep this in mind if you
   modify the game settings.
3. If you have a 4k screen and want to capture 4k data (very hardware hungry,
   but runs smooth on an RTX3090): Set the screen resolution to "7680x4320DSR"
   in NVIDIA GeForce Experience. This increases the buffer sizes for pixel
   perfect 4k segmentation data.
4. Set the correct screen Resolution in `VPilot/utils/Constants.py`. To do this
   change the variable `MANAGED_SCREEN_RESOLUTION` to the screen resolution at
   which GTAV is running, e.g. "1920x1080". Default is "3840x2160DSR7680x4320"

## Data Generation using VPilot
VPilot uses simple Python commands to interact with DeepGTAV. Examples of how
VPilot can be used are shown in `VPilot/presentation_VisDrone.py`. 

The other scripts in the `VPilot` folder starting with `datageneration_` can be
used to generate different synthetic training data matching different real world
datasets.

To do data generation simply run GTAV. When the game has loaded run one of the
respective data generation scripts and specify the Export directory, e.g.:

      python .\VPilot\datageneration_VisDrone.py --save_dir "C:\DeepGTAV_Exportdir\"

Then open the GTAV window again, ESC from the menu and the data generation
should be starting.

In some rare cases the game crashes when the player is in a building while
starting a data generation script. To prevent this leave the building before
starting the data generation.


ATTENTION: Restrict the data that is recorded with the `Dataset` message to only
the data that you need. 

e.g.

      dataset=Dataset(location=True, time=True, exportBBox2D=True)

against

      dataset=Dataset(location=True, time=True, exportBBox2D=True, segmentationImage=True, exportLiDAR=True, maxLidarDist=120)

This improves the capturing speed by quite a lot.



In `VPilot/backup/presentation_VisDrone_VisualizeMessageParts.py` examples on
how to handle different message parts are given (most in comments). Note that
most of the export messages are only there for legacy and debugging reasons. The
suported and useful exported message parts are the settings `exportBBox2D,
segmentationImage, exportLiDAR (with maxLidarDist)` in `Dataset` and
corresponding message parts. 



### VPilot messages
For an in depth understanding of the VPilot message interface, the best way to
start is to look at the file `VPilot\deepgtav\messages.py`. The function
signatures of the `__init__` should be realtively self explanatory.


DeepGTAV sends capturing messages to VPilot in every tick. Those messages encode
all the recorded information and can be accessed by their fields (e.g.
message['frame'] is the recorded image, message['bbox2d'] the recorded bounding
boxes etc.)


Note that there are lots of legacy fields in the `Dataset` message, which
sometimes do nothing. I would advise to only use the fields that are also used
in the example scripts and data generation scripts.


### Capturing images freely from arbitrary locations
If you want to capture images from any location a hacky simple solution is to use a 
"freecam" mod to move the camera. [Aikido Free Cam](https://www.gta5-mods.com/scripts/aikido-free-cam) 
worked well for us (see this [issue](https://github.com/Eisbaer8/DeepGTAV/issues/7)).


## Modification of GTAV
In general DeepGTAV should work stable with modifications of GTAV. 

For my works I used the following modifications:
1. Simple Increase Traffic(and Pedestrian)
2. HeapLimitAdjuster
3. Balanced Classes
4. To allow 4k Data capturing the resolution has to be set to 7680x4320DSR in
   NVIDIA GeForce Experience. This correctly scales the Depth and Stencil
   buffers which are used to determine object segmentation. 

The modifications and install instructions can be found in the folder `Mods`

Different additional modifications could be used, mainly for graphics
improvements. If you use this repository in conjunction with graphics enhancing
mods I would love it if you send me a quick email telling me what mods you used
and whether they worked. 






# In Depth Use
## General Module Structure
This framework consists of different moving parts, that are organized in
different folders. Their interaction is described in the following:

`DeepGTAV-PreSIL` contains the code to build the scripts that start the DeepGTAV
server in GTAV. Communication with those scripts is done via a TCP-Port. In
`VPilot` the functionalities to communicate with DeepGTAV are implemented, to
control DeepGTAV from a Python script. 

`DeepGTAV-PreSIL` uses the functionalities implemented in
`GTAVisionExport-DepthExtractor` to extract depth and stencil information. To
communicate with GTAV DeepGTAV uses `SkriptHookV`. 

In the `Mods/Readme.md`, installation instructions for different mods that were
used to improve GTAV are shown. Mainly those are changes to spawn more objects
in the game world (e.g. cars and pedestrians) but graphics enhancements will be
added in the future.



In depth explanations can be found in bachelors thesis. In the future I will try
to do better documentation, but for now this has to suffice.





# Development
If you want to compile the binaries by yourself or make your own changes you
have to do the following things:

DeepGTAV was built using MS Visual Studio 2017 and I recommend to also use it to
circumvent compatability issues.

- [ ] TODO this is currently not up to date!

1. clone this repository
2. TODO get all requirements for DeepGTAV-PreSIL and copy them to this
   directory, in particular those are: 
   1. GTAVisionExport-DepthExtractor (already contained in this repository)
   2. eigen-3.3.7 (https://eigen.tuxfamily.org/index.php?title=Main_Page get the
      source of version 3.3.7)
   3. opencv343 (https://opencv.org/releases/ get the source of version 3.4.3)
   4. boost_1_73_0 (https://www.boost.org/users/history/version_1_73_0.html get
      the source of verions 1.73.0)
   5. zeroMQ TODO 

   Build GTAVisionExport-DepthExtractor and zeroMQ. Specific build instructions can be found in the individual projects.
   In principle it suffices to build using cmake and then building them in Visual Studio. A detailed description is as follows:

   4. Run cmake (cmake-gui) from your Windows start menu.
   5. Hit 'Browse Source' and select your GTAVisionExport/native folder.
   6. Hit 'Browse Build', create GTAVisionExport/native/build folder and select it.
   7. Hit 'configure' (first time around it will fail but dont worry).
   8. Choose project generator 'Visual Studio 15 2017 Win64' and keep the option 'use default native compilers'
   9. After the fail dialog, modify the EIGEN3_INCLUDE_DIR to point to your Eigen3 folder.
   10. Run 'configure' followed by 'generate'.
   11. cmake should now have generated the Visual Studio solution into GTAVisionExport/build.
   12. Open 'GTANativePlugin.sln' in Visual Studio.
   13. Select 'release' from the 'Solution Configurations' drop down.
   14. Edit GTAVisionNative project properties/configuration properties/c/c++/additional include dirs in VS to add the GTAVisionExport/native/src folder (this allows VS to find MinHook.h)
   15. Edit GTAVisionNative project properties/configuration properties/linker/input/additional dependencies to add : 
   `"..\..\deps\libMinHook.x64.lib"` 
   16. Press F6 to build the solution. it should now succeed and the products should be in 'GTAVisionExport\native\build\src\Release'
   17. Copy GTAVisionNative.asi & GTAVisionNative.lib to your GTAV exe folder.


   It is important to mimic this exact file structure, otherwise the relative
   "Additional Include Directories" do not fit correctly. If this is the case
   and for some reason you want to include those directories from different
   locations, you have to change the "Additional Include Directories" in the
   following way: 
   
   Right click DeepGTAV in the SolutionExplorer in Visual Studio -> Properties
   -> C/C++ -> Additional Include Directories

   Here the include directories have to match.

   Furthermore in 

   Properties -> Linker -> Input -> Additional Dependencies 

   the correct locations of the libraries have to be set 

3. Build GTAVisionExport-DepthExtractor as described. Copy the files from
   `GTAVisionExport-DepthExtractor\native\build\src\Release` to
   `DeepGTAV-PreSIL\bin\Release` (This is not necessary, if DepthExtractor has
   not been changes).



4. Set environment variable `GTAV_INSTALL_DIR` to point to the root directory of
   GTAV. This is done so that on building DeepGTAV-PreSIL the newly built files
   are automatically copied into the GTAV install directory.
5. Set environment variables `DEEPGTAV_EXPORT_DIR` and `DEEPGTAV_LOG_FILE`
   (those are legacy requirements and should be removed in future versions)






# Repository History
This repository is built upon different other works. If you use this repository
please cite those respective works and this repository.

The initial DeepGTAV framework was built in this repo:
https://github.com/aitorzip/DeepGTAV


In the work "Driving in the Matrix: Can Virtual Worlds Replace Human-Generated
Annotations for Real World Tasks?" (https://arxiv.org/abs/1610.01983) object
segmentation exportation from GTAV was provided:
https://github.com/umautobots/GTAVisionExport

In "Precise Synthetic Image and LiDAR (PreSIL) Dataset for Autonomous Vehicle
Perception" (https://arxiv.org/abs/1905.00160) those two functionalities were
combined: https://github.com/bradenhurl/DeepGTAV-PreSIL


Primarily this repository contains changes that were made to allow the capturing
of data from UAV perspective. Additionally some quality of life changes have
been made, a comprehensive list of the improvements is given in the following:

- [x] The TCP server functionality has been replaced with ZeroMQ for more stability
  and speed.
- [x] All data export functionality by writing to disk has been replaced by sending
  this data over the TCP socket to Python. This increased capturing speeds and
  usability.
- [x] Multiple commands have been added to VPilot to allow fine grained control of
  what data is being captured and of the game world (e.g. spawning pedestrians,
  moving to ingame locations, changing the camera perspective).
- [x] The code quality has been improved, some bugs were fixed and some functions
  have been refactored for more efficiency. 
- [x] The total capturing speed has been improved by a factor of 3-4, resulting in
  almost no overhead to running the game natively.
- [x] Different skripts are provided to capture data in vastly different scenarios. 
- [x] Compatability with the newest game version has been tested.
- [x] The installation of DeepGTAV to the game files has been simplified
<!-- A detailed description of the changes is presented in my bachelors thesis.  -->


# Legals and Use Considerations
In principal, the publisher of GTAV (Rockstar Games) allows for non-commercial use 
of game footage (http://tinyurl.com/pjfoqo5), which includes academic work.
Commercial use of the game footage however is not allowed in general. 

Furthermore when working or extending this tool it has to be considered that 
new game patches could introduce breaking changes at any time. Note however that 
this has not happened since the release of the game, so this is rather unlikely.



# TODO 
## Improvements
- [x] Implement object spawning 
- [x] increase amount of spawned objects 
- [x] balance spawned object classes

- [x] Adapt for higher resolution capturing 


## Things that I will not do in the near future that would be easy to implement or useful
- [ ] Allow capturing of different objects (Traffic signs, Animals, houses,...)
- [ ] Improve the graphics quality 
- [ ] Improve the water quality
- [ ] Improve the segmentation quality on water (see below) 
- [ ] Sending the ground truth 3D bounding boxes from DeepGTAV to VPilot
- [ ] Compress messages sent through ZeroMQ
- [ ] The capturing process could be made to feel smoother by attaching the
  camera to the entity and not setting the camera position relative to the
  entity at each captured frame. This would reduce the stutter (but does not
  change captured data quality).



## Things to note
- The LiDAR simulation currently uses the ingame raycasting and the depth map
  from the graphics. For the purposes of the ingame ray casting low poly objects
  are used (e.g. tree tops are only balls). The cast rays are then updated with
  the correct depth from the graphics depth map, if it is available. Of course
  this depth map is only available for the objects that are in the current
  frame. Therefore the LiDAR is not exact for objects outside of the current
  frame image. 
- Currently the calcualtion of ingame positions from the depth map
  and the determination of the correct object segmentation from this data is
  done relatively inefficient on the CPU. Great performance improvements could
  be achieved by doing this on the GPU.
- Currently the segmentation of objects on water is relatively unintuitive. The
  segmentation works in such a way that only the part of the object above the
  water surface is part of the segmentation mask, while the part of the object
  below the water surface is not part of the segmentation mask. However parts of
  the objects below the water are still visible. With our current methodolgy of
  calculating the segmentation data from the depth and stencil buffers it is not
  possible to extend the segmentation to the parts of the objects below the
  water surface. However, if one wanted to fix this, there is a water opacity
  buffer in the rendering pipeline that contains inforamtion about the depth
  below the water for each pixel (see
  https://www.adriancourreges.com/blog/2015/11/02/gta-v-graphics-study/ for a
  detailed explanation of the GTAV rendering pipeline, you can use RenderDoc or
  NVIDIA Nsight to dissect the rendering pipeline). This buffer would need to be
  extracted by extending GTAVisionExport-DepthExtractor. Then one could combine
  this buffer data with the stencil buffer and then get the correct segmentation
  masks from those. 
- If too many messages are sent over the ZeroMQ mesage queue in short intervals,
  or if the sent messages are too large (e.g. extracting multiple images +
  LiDAR) then the message queue can get overfilled, which results in messages
  not being processed in the correct order (e.g. a `StopCapture` message does
  not come through and too many captures are sent for some frames). If this
  happens increase the spacing between captures, or reduce the amount of
  captured data. 
   - Another more advanced fix would be to compress the sent messages.



