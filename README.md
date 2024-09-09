<!-- PROJECT LOGO -->
<br />
<p align="center">

  <h3 align="center">Game4Loc: A UAV Geo-Localization Benchmark from Game Data</h3>

</p>



![](resources/GTA-UAV-data-construction.jpg)
*GTA-UAV data construction*


## GTA-UAV Dataset Structure

```
├─ GTA-UAV
|   ├── drone/
|   |   ├──images/
|   |   |   ├── 100_0001_0000000001.png
|   |   |   ├── 200_0001_0000000001.png
|   |   |   └── ...
|   |   └──meta_data/
|   |       ├── 100_0001_0000000001.txt
|   |       ├── 200_0001_0000000001.txt
|   |       └── ...
|   ├── cross-area-drone2sate-train.json
|   ├── cross-area-drone2sate-test.json
|   ├── same-area-drone2sate-train.json
|   └── same-area-drone2sate-test.json

```


