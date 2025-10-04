# Person Segmentation in Videos using DETR + SAM 2

## Overview

This project automatically detects and segments **people in video frames** using two state-of-the-art models from the Hugging Face ecosystem:

- **[DETR (ResNet-50)](https://huggingface.co/facebook/detr-resnet-50)** – detects bounding boxes for persons.  
- **[SAM 2 Hiera](https://huggingface.co/facebook/sam2-hiera-large)** – refines bounding boxes into pixel-accurate segmentation masks.

The tool processes a video, replacing detected person regions with distinct solid colors while leaving the background unchanged.  
It’s ideal for **semantic video preprocessing**, **privacy masking**, or **dataset analysis**.

---

## Features

- Uses **local pretrained models** if provided, or **downloads them automatically**.  
- GPU acceleration via PyTorch CUDA (falls back to CPU if unavailable).  
- Batch-wise frame processing to control memory usage.  
- Generates masked output videos with suffix `_person_segments.mp4`.  

---

## Example Usage

### Process a video with local models

```bash
python person_segments.py --input "D:\Videos\clip.mp4"  --detr-dir "D:\DeTR\Model" --sam2-dir "D:\SAM\Modelv2" --out-dir "D:\Videos\out"
