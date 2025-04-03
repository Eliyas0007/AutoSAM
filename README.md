# AutoSAM2
Automated [SAM2](https://github.com/facebookresearch/sam2) segmentation and tracking primarily for videos

## Overview
This repo uses YOLOv11 as an prompt generator to let sam2 work on these prompts (bounding boxes of objects) to segment and track objects in videos. We will integrate CLIP to support text prompts in the future.

## Setup
Download the repo:
```
git clone https://github.com/Eliyas0007/AutoSAM.git
cd AutoSAM
```
Install the requiered packages and create new virtual environment using conda:
```
conda env create -f environment.yml
conda activate SAM2
```
Install SAM2 with following:
```
cd sam2
pip install -e .
cd ..
```
Download the checkpoints for SAM2 and YOLOv11:
```
cd chackpoints
bash download_ckpts.sh
```
Now you have successfully installed SAM2 and YOLOv11, if you have any issues please refer to the original instruction [here for SAM2](https://github.com/facebookresearch/sam2) and [here for YOLOv11](https://github.com/ultralytics/ultralytics)

## How to run
You can check for each of the arguments and how to use them in```scripts/segment_videos.py```from main function. Each argument has already set to a default value but you still need to change them according to your needs. Then you can run a demo segmentation of a video with command in a terminal as follows:
```
python scripts/segment_videos.py --detect_first_frame_only --is_demo
```
## Data
You can check path```dummy_video_dataset/``` to prepare a dataset with similar structure.
- First a root directory for your dataset
- Then each video is stored as a folder inside the root dir
- Each folder stores the frames of this video either as ```.png``` or ```.jpg```/```.jpej```

## TODO
Integrate CLIP to provide alternative segmentation method which has dynamic object detection

## Acknowledgements
The demo dataset we used in this repository is [flat'n'fold dataset](https://arxiv.org/abs/2409.18297)



