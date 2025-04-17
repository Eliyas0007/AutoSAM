# AutoSAM2

**Automated SAM2-based segmentation and tracking for video analysis**

AutoSAM2 enhances [SAM2](https://github.com/facebookresearch/sam2) by automatically generating segmentation prompts using other pretrained models such as YOLOv11 and CLIPSeg. These prompts (bounding boxes or masks) enable SAM2 to perform accurate object segmentation and tracking in videos.

With CLIPSeg integration, you can now use **text prompts** to guide segmentation — enabling more flexible and semantic-aware object detection.

---

## 🚀 Features
- 🔍 Automatically generate prompts using YOLOv11, CLIPSeg
- 📝 Use **text-based prompts** (via CLIPSeg) for semantic object segmentation  
- 🎥 Segment and track objects across video frames using SAM2  
- ⚡️ Detect objects on the first frame only for faster inference  
- 📁 Easy-to-use demo dataset and command-line interface  

---

## 🛠️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/Eliyas0007/AutoSAM.git
cd AutoSAM
```

### 2. Create and activate the conda environment
```bash
conda env create -f environment.yml
conda activate SAM2
```

### 3. Install SAM2
```bash
cd sam2
pip install -e .
cd ..
```

### 4. Download checkpoints
```bash
cd chackpoints
bash download_ckpts.sh
```

> ✅ If you encounter any issues, refer to the original instructions for [SAM2](https://github.com/facebookresearch/sam2) and [YOLOv11](https://github.com/ultralytics/ultralytics).

---

## 📼 How to Run

You can run a demo segmentation using:

```bash
python scripts/segment_videos.py --detect_first_frame_only --is_demo
```
To use CLIPSeg, simply add ```--use_clipseg``` as an additional flag like follows:
```bash
python scripts/segment_videos.py --detect_first_frame_only --is_demo --use_seg
```

To explore available arguments, check the `main()` function in [`scripts/segment_videos.py`](scripts/segment_videos.py). Most parameters have default values, but you should modify paths or settings based on your dataset and use case.

---

## 📂 Dataset Structure

Prepare your video dataset similar to the provided `dummy_video_dataset/` directory:

```
your_dataset_root/
├── video_1/
│   ├── frame_0001.jpg
│   ├── frame_0002.jpg
│   └── ...
├── video_2/
│   ├── frame_0001.jpg
│   ├── ...
```

- Each **video** is a folder inside the dataset root.
- Each folder contains **frames** of the video as `.png`, `.jpg`, or `.jpeg` files.

## 📤 Output Structure

After running the segmentation script, the output will be organized as follows. Each video folder will contain:

- Subfolders for each segmented **instance/object** (e.g., `object1`, `object2`, ...), each containing the extracted frames where that object appears.
- A separate `background/` folder that stores the remaining parts of the frames with the objects removed (optional, depending on settings).

```
your_dataset_root/
├── video_1/
│   ├── object1/
│   │   ├── frame_0001.png
│   │   ├── frame_0002.png
│   │   └── ...
│   ├── object2/
│   │   ├── frame_0001.png
│   │   ├── ...
│   ├── background/
│   │   ├── frame_0001.png
│   │   └── ...
│   └── ...
├── video_2/
│   ├── object1/
│   ├── object2/
│   ├── background/
│   └── ...
```

- `object1/`, `object2/`, etc. represent individual segmented objects.
- `background/` contains frames with the segmented objects removed or masked out.
- This structure makes it easy to analyze or visualize each object separately from the background.
- You can use the segmented videos as a dataset to train [SCAT](https://github.com/Eliyas0007/SCAT)

---

## ✅ TODO
- [ ] Code refactoring  
- [x] Integrate CLIPSeg for dynamic, text-based object detection 
---

## 📚 Acknowledgements

- The demo dataset used is from [Flat’n’Fold: A Dataset for Flat Object Manipulation](https://arxiv.org/abs/2409.18297)  
- Thanks to [Facebook Research](https://github.com/facebookresearch/sam2), [Ultralytics](https://github.com/ultralytics/ultralytics) and [HuggingFace](https://huggingface.co/docs/transformers/en/model_doc/clipsega) for the original models

---
