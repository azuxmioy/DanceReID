# DanceReID

## Prerequisites


```
apt-get install youtube-dl
pip3 install torch Pillow opencv-python tqdm numpy
```

## Data preparation

### Download raw bounding boxes detections and human pose landmarks npy files

[Google drive](https://drive.google.com/file/d/1qSh78lGQ6b7bZsg8K3tG1SIWaVSjoUrc/view?usp=sharing) (237MB)


### Video crawler

Download the selected youtube videos using the following command:
```shell
bash run.sh /path/to/video_folder video_data.csv
```

### DanceReID dataset generation

Generate image-based dataset for re-ID using the following script:
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
|-- poses/......................( if using --split-folder flag)
|   |-- video_folders/ 
|--  skeleton/ .................( if using -gs flag)
|   |-- video_folders/ .........( if using --split-folder flag)
|-- splits.json 
|-- meta.json
|-- video.json
|-- DanceReID.h5 ...............( generated if using -h5 flag)
```
Note that if you did not apply the --split-folder flag when generating data, there will be no separated video folders.

### Dataset statistic

| subset   | # ids |# images| # videos |
| -------- | ------| ------ |  ------  | 
| trainval |   71  |  41242 |    15   | 
| test     |   29  |  21356 |     6    |

Evaluation metric: **mAP**, **CMC-CUHK03** (single gallery shot)
Please find more details for every single video in the csv file.


## Baseline performance

### Simple Baseline w/ ResNet-50 backbone
| Tricks  | mAP  | rank-1 | rank-5 |
| ------  |----  | ------ | ------ |
| Softmax | 74.4 |  73.1  |  94.7  |
| Siamese | 77.5 |  75.9  |  96.9  |
| Triplet | 78.4 |  77.2  |  97.6  |
