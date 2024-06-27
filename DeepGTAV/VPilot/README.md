# VPilot
Scripts and tools to easily communicate with DeepGTAV. 

## How it works

VPilot uses JSON over TCP sockets to start, configure and send/receive commands
to/from DeepGTAV by using the Python DeepGTAV libraries. 



## Scripts in this directory
`drive_FOR_PRESENTATION.py` shows most of the possible commands and was used to
capture a demo of the functionality. `drive_LOW_USED_GENERATION.py` and
`drive_NO_IMPROVEMENTS(10-100m).py` are the two scripts that were used to
generate the datasets DeepGTAV_LOW and DeepGTAV_HIGH respectively, that were
used for training in my Bachelors Thesis. 

`visualizeGeneratedExportDir.py` can be used to visualize the exported directory
with bounding boxes. 


_dataset_ and _drive_ are legacy files that were used to collect data in the
initial version of DeepGTAV. 
