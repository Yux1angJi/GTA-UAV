<!-- PROJECT LOGO -->
<br />
<p align="center">

  <h3 align="center">Game4Loc: A UAV Geo-Localization Benchmark from Game Data</h3>

</p>


<p align="center">
  <img src="resources/trajectory_demo_compress.gif" alt="Demo">
  <br>
  <i>
  Localization in flight trajectory after pre-trained on GTA-UAV dataset.
  </i>
</p>


<p align="center">
  <a href="https://yux1angji.github.io/" target='_blank'>Yuxiang Ji*</a>,&nbsp;
  Boyong He*,&nbsp; Zhuoyue Tan,&nbsp; Liaoni Wu
  <br>
  <a href="https://yux1angji.github.io/game4loc/"><strong>Project Page »</strong></a>
</p>

- [x] Part I: Dataset
- [x] Part II: Train and Test
- [ ] Part III: Pre-trained Checkpoints

## Table of contents

- [Dataset Highlights](#dataset-highlights)
- [Dataset Access](#dataset-access)
- [Dataset Structure](#dataset-structure)
- [Train and Test](#train-and-test)
- [Pre-trained Checkpoints](#pre-trained-checkpoints)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)


## Dataset Highlights
![](resources/GTA-UAV-data-construction.jpg)
*GTA-UAV data construction*

<b><i>GTA-UAV</i> dataset</b> provides a large continuous area dataset (covering 81.3km<sup>2</sup>) for UAV visual geo-localization, expanding the previously aligned drone-satellite pairs to **arbitrary drone-satellite pairs** to better align with real-world application scenarios. Our dataset contains:

- 33,763 simulated drone-view images, from multiple altitudes (80-650m), multiple attitudes, multiple scenes (urban, mountain, seaside, forest, etc.).

- 14,640 tiled satellite-view images from 4 zoom levels for arbitrarily pairing.

- Overlap (in IoU) of FoV for each drone-satellite pair.

- Drone (camera) 6-DoF labels for each drone image.

## Dataset Access
The dataset is released in two versions: low resolution (512x384, 12.8G) and high resolution (1920x1440, 133.6G).

|                                      Low Resolution Version                                      |                                     High Resolution Version                                      |
|:------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------:|
| [HuggingFace🤗](https://huggingface.co/datasets/Yux1ang/GTA-UAV-LR) | Released soon |
| [BaiduDisk](https://pan.baidu.com/s/1WyadL67Wxmij2be3qjrihg?pwd=gtav) | Released soon |


The high resolution dataset will be released soon.

## Dataset Structure

### Directory Structure
```
├─ GTA-UAV
|   ├── drone/
|   |   └── images/
|   |       ├── 200_0001_0000000001.png
|   |       ├── 200_0001_0000000002.png
|   |       └── ...
|   ├── satellite/
|   |   ├── 6_0_0_0.png
|   |   ├── 6_0_0_1.png
|   |   └── ...
|   ├── cross-area-drone2sate-train.json
|   ├── cross-area-drone2sate-test.json
|   ├── same-area-drone2sate-train.json
└── └── same-area-drone2sate-test.json
```

### Example Entry in `x-area-drone2sate-x.json`

This entry provides a detailed description and paired satellite images for a single drone image in the training/test dataset:

```json
{
    "drone_img_dir": "drone/images",
    "drone_img_name": "400_0001_0000019853.png",
    "drone_loc_x_y": [
        4398.2345699437265, 7743.059007970046
    ],
    "sate_img_dir": "satellite",
    "pair_pos_sate_img_list": [
        "5_0_12_22.png"
    ],
    "pair_pos_sate_weight_list": [
        0.4923079286001371
    ],
    "pair_pos_sate_loc_x_y_list": [
        [4320.0, 7776.0]
    ],
    "pair_pos_semipos_sate_img_list": [
        "4_0_6_11.png",
        "5_0_12_22.png",
        "6_0_25_44.png",
        "6_0_25_45.png"
    ],
    "pair_pos_semipos_sate_weight_list": [
        0.16504440259875908,
        0.4923079286001371,
        0.33077424066935046,
        0.24444984927508234
    ],
    "pair_pos_semipos_sate_loc_x_y_list": [
        [4492.8, 7948.8],
        [4320.0, 7776.0],
        [4406.4, 7689.6],
        [4406.4, 7862.4]
    ],
    "drone_metadata": {
        "height": 405.9933166503906,
        "drone_roll": -1.237579345703125,
        "drone_pitch": 3.1808385848999023,
        "drone_yaw": -122.0,
        "cam_roll": -91.23757934570312,
        "cam_pitch": 3.1808385848999023,
        "cam_yaw": -122.0
    }
}
```

### Metadata Details

- `drone_loc_x_y`: Provides the 2D location for the centre of drone-view image.

- `pair_pos_sate_img(weight/loc_x_y)_list`: Provides the positive paired satellite image / weight(IOU) / 2D location list.

- `pair_pos_semipos_sate_img(weight/loc_x_y)_list`: Provides the semi-positive paired satellite image / weight(IOU) / 2D location list.

- `drone_metadata`: Provides the height (altitude above ground level), drone pose (roll, pitch, yaw), and camera pose (roll, pitch, yaw) information.


### Collect Your Own Data

You may want to collect your own data from simulated game environments, if so, you could refer [here](DeepGTAV/VPilot/datageneration_GeoLoc.py).

To configure the simulation and collection environment, please refer [DeepGTA](https://github.com/David0tt/DeepGTAV).
Notice that the compiled `DeepGTA` plugin for our GTA-UAV data simulation and collection is located at [here](DeepGTAV/DeepGTAV-PreSIL/bin/Release/).

## Train and Test

![](resources/pipeline.jpg)
*Proposed training and test pipeline*

First, install dependencies   
```bash
cd Game4Loc
# install project   
pip install -e .   
pip install -r requirements.txt
```

Then you could simply run the training experiments by
```bash
# run experiment (example: GTA-UAV cross-area setting)  
python train_gta.py \
    --data_root <The directory of the GTA-UAV dataset> \
    --train_pairs_meta_file "cross-area-drone2sate-train.json" \
    --test_pairs_meta_file "cross-area-drone2sate-test.json" \
    --model "vit_base_patch16_rope_reg1_gap_256.sbb_in1k" \
    --gpu_ids 0,1 --label_smoothing 0.05 \
    --lr 0.0001 --batch_size 64 --epoch 5 \
    --with_weight --k 5
```

## Pre-trained Checkpoints
To be released soon.

## License
This project is licensed under the [Apache 2.0 license](LICENSE).


## Acknowledgments 
This work draws inspiration from the following code as references. We extend our gratitude to these remarkable contributions:

- [Sample4Geo](https://github.com/Skyy93/Sample4Geo)
- [DeepGTA](https://github.com/David0tt/DeepGTAV)
- [GTA-V-Wolrd-Map](https://github.com/Flamm64/GTA-V-World-Map)

## Citation
To be released soon.
