<!-- PROJECT LOGO -->
<br />
<p align="center">

  <h3 align="center">Game4Loc: A UAV Geo-Localization Benchmark from Game Data</h3>

</p>

<div style="text-align: center;">
    <video width="600" controls>
    <source src="resources/trajectory_demo_compress.mp4" type="video/mp4">
    Your browser does not support the video tag.
    </video>
    <p style="margin-top: 10px;">Localization in flight trajectory after pre-trained on GTA-UAV dataset.</p>

<p align="center">
    <a href="https://yux1angji.github.io/game4loc/"><strong>Project Page »</strong></a>
</p>
</div>

- [x] Part I: Dataset
- [x] Part II: Train and Test
- [ ] Part III: Pre-trained Checkpoints

## Table of contents

- [Dataset Highlights](#dataset-highlights)
- [Dataset Access](#dataset-access)
- [Dataset Structure](#dataset-structure)
- [Train and Test](#train-and-test)
- [Pre-trained Models](#pre-trained-models)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)


## Dataset Highlights
![](resources/GTA-UAV-data-construction.jpg)
*GTA-UAV data construction*

<b><i>GTA-UAV</i> dataset</b> provides a large continuous area dataset (covering 81.3km<sup>2</sup>) for UAV visual geo-localization, expanding the previously aligned drone-satellite pairs to **arbitrary drone-satellite pairs** to better align with real-world application scenarios. Our dataset contains:

- 33,763 simulated drone-view images, from multiple altitudes (100-650m), multiple attitudes, multiple scenes (urban, mountain, seaside, forest, etc.).

- 14,640 tiled satellite-view images from 4 zoom levels for arbitrarily pairing.

- IoU of FoV for each drone-satellite pair.

- Drone (camera) 6-DoF labels for each drone image.

## Dataset Access
The low resolution dataset (13.4G) is released in [HuggingFace🤗](https://huggingface.co/datasets/Yux1ang/GTA-UAV-LR).

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
|   |   ├── 6_0_0.png
|   |   ├── 6_0_1.png
|   |   └── ...
|   ├── cross-area-drone2sate-train.json
|   ├── cross-area-drone2sate-test.json
|   ├── same-area-drone2sate-train.json
|   └── same-area-drone2sate-test.json
```

### Example Entry in `x-area-drone2sate-x.json`

This entry provides a detailed description and paired satellite images for a single drone image in the training/test dataset:

```json
{
  "drone_img_dir": "drone/images",
  "drone_img_name": "500_0001_0000024847.png",
  "drone_loc_x_y": [
      3633.472345321685,
      7140.565493165591
  ],
  "sate_img_dir": "satellite",
  "pair_pos_sate_img_list": [
      "5_10_20.png"
  ],
  "pair_pos_sate_weight_list": [
      0.5786491782179856
  ],
  "pair_pos_sate_loc_x_y_list": [
      [
          3628.8,
          7084.8
      ]
  ],
  "pair_pos_semipos_sate_img_list": [
      "4_5_10.png",
      "5_10_20.png",
      "6_20_41.png",
      "6_21_40.png",
      "6_21_41.png"
  ],
  "pair_pos_semipos_sate_weight_list": [
      0.25622919559581886,
      0.5786491782179856,
      0.2188736257558427,
      0.19654190179816708,
      0.19387413912296475
  ],
  "pair_pos_semipos_sate_loc_x_y_list": [
      [
          3801.6,
          7257.6
      ],
      [
          3628.8,
          7084.8
      ],
      [
          3542.4,
          7171.2
      ],
      [
          3715.2000000000003,
          6998.400000000001
      ],
      [
          3715.2000000000003,
          7171.2
      ]
  ],
  "drone_metadata": {
      "height": 501.8125,
      "drone_roll": 4.607879638671875,
      "drone_pitch": 2.0246360301971436,
      "drone_yaw": 40.999996185302734,
      "cam_roll": -85.39212036132812,
      "cam_pitch": 2.0246360301971436,
      "cam_yaw": 40.999996185302734
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
Notice that the compiled `DeepGTA` plugin for our GTA-UAV data simulation is located at [here](DeepGTAV/DeepGTAV-PreSIL/bin/Release/).

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
python train_gta.py --data_root <The directory of the GTA-UAV dataset> --train_pairs_meta_file "cross-area-drone2sate-train.json" --test_pairs_meta_file "cross-area-drone2sate-test.json" --gpu_ids 0,1 --label_smoothing 0.05 --with_weight --k 5 --epoch 5 --model 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k' --lr 0.0001 --batch_size 64
```

## Pre-trained Models
To be released soon.

## License
This project is licensed under the [Apache 2.0 license](LICENSE).


## Acknowledgments 
This work draws inspiration from the following code as references. We extend our gratitude to these remarkable contributions:

- [Sample4Geo](https://github.com/Skyy93/Sample4Geo)
- [DeepGTA](https://github.com/David0tt/DeepGTAV)
- [GTA-V-Wolrd-Map](https://github.com/Flamm64/GTA-V-World-Map)

## Citation

