# DanceReID

![](https://i.imgur.com/TAB996D.jpg)

This repository provides the necessary codes and scripts for preparing our Dance ReID dataset. The videos in the dataset are collected from YouTube Creative Commons. We then perform human pose tracking for each video to extract the human bounding boxes and skeleton landmarks. Finally, we manually filter out label misalignment of the bounding boxes and build up a large-scale dataset for the multi-person tracking re-ID purpose.

Our dataset includes:
* Cropped bounding boxes with human ID labels
* Human pose landmarks for every bounding box (in pixel coordinate)
* Diverse human pose for each person (also with their skeleton rendering map)

## Prerequisites

We use [youtube-dl](https://github.com/ytdl-org/youtube-dl) to automatically download the selected videos. Make sure you are using the up-to-date version.
```
apt-get install youtube-dl
```
We implement our code in Python3, please install the following packages
```
pip3 install Pillow opencv-python tqdm numpy h5py
```
## Data Preparation

### Download raw bounding boxes detections and human pose landmarks npy files

[Google drive](https://drive.google.com/file/d/1Ph8yUuXEefU6Buv1dOrxXFFDuptyEdA2/view?usp=sharing) (237MB)


### Video crawler

Download the selected youtube videos using the following command:
```shell
bash run.sh /path/to/video_folder video_data.csv
```

### DanceReID dataset generation

Generate an image-based dataset for re-ID using the following script:
```shell
python3 gen_DanceReID.py -i /path/to/video_folder -n /path/to/npy_folder 
        -a /path/to/annotation_json -d 5 [ -o /path/to/output_folder ] [ -gs ] 
        [ --split-folder ] [ -h5 ]
```

The resulting dataset folder should have the structure as below:
```
path/to/your/DanceReID/
|-- images/.....................( if using --split-folder flag)
|   |-- video_folders/ 
|        ...
|-- poses/......................( if using --split-folder flag)
|   |-- video_folders/ 
|        ...
|--  skeleton/ .................( if using -gs flag)
|   |-- video_folders/ .........( if using --split-folder flag)
|        ...
|-- splits.json 
|-- meta.json
|-- video.json
|-- DanceReID.h5 ...............( generated if using -h5 flag)
```
Note that if you did not apply the --split-folder flag when generating data, there will be no separate video folders.


## Baseline performance

### Dataset statistic

In our paper, we downsample the videos every 5 frames(using the tag -d 5) for evaluation. This results in the following dataset statistic

| subset   | # ids |# images| # videos |
| -------- | ------| ------ |  ------  | 
| trainval |   71  |  31643 |    15   | 
| test     |   29  |  19526 |     6    |

Please find more details for every single video in the csv file. 

**Note:** In our paper, we use only 100 IDs for the experiments. Here we also provide another version of our dataset (a total of 178 IDs in 33 videos) for further research, you can download all videos by replacing [this csv file](https://drive.google.com/file/d/1egXS4Wca-LrxExGRG6A3emXMIReh6-ZR/view?usp=sharing).

### Simple Baseline w/ ResNet-50 backbone

Evaluation metric: **mAP**, **CMC-CUHK03** (single gallery shot)

| Methods    | mAP  | rank-1 | rank-5 |
| ------    |----  | ------ | ------ |
| Softmax [[xiao2016]](https://arxiv.org/abs/1604.07528) | 74.4 |  73.1  |  94.7  |
| Siamese [[chung2017]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Chung_A_Two_Stream_ICCV_2017_paper.pdf)| 77.5 |  75.9  |  96.9  |
| Triplet [[hermans2017]](https://arxiv.org/abs/1703.07737)| 78.4 |  77.2  |  97.6  |
| ST-ReIDNet | 86.1 |  84.9  |  98.7  |

Check our baseline and model implementation in the [ST-ReIDNet](https://github.com/azuxmioy/ST-ReIDNet) repository.
