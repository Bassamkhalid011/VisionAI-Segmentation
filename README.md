# VisionAI Segmentation Studio
 
An AI-powered image segmentation web app built with **YOLOv8 + Meta SAM (Segment Anything Model)**. Upload any image and get intelligent object detection and segmentation with multiple output modes.
 
---
 
## Features
 
- Real-time object detection using **YOLOv8**
- Precise segmentation masks using **Meta SAM**
- 3 output modes:
  - **Segmentation Overlay** — colorful masks over detected objects
  - **Blur Background** — keeps subject sharp, blurs background
  - **Remove Background** — transparent background PNG output
- Displays object count, processing time, and confidence scores
- Download processed result image
- GPU (CUDA) support for faster processing
---
 
## Prerequisites
 
| Requirement | Notes |
|---|---|
| Python 3.8+ | |
| CUDA (optional) | For GPU acceleration |
| SAM Model Checkpoint | Download from Meta |
| YOLOv8 | Auto-downloaded on first run |
 
---
 
## Setup
 
### 1. Install dependencies
 
```bash
pip install streamlit opencv-python pillow numpy torch ultralytics segment-anything
```
 
### 2. Download SAM model checkpoint
 
Download `sam_vit_b_01ec64.pth` from Meta's official SAM repo and place it in a `models/` folder:
 
```
models/
└── sam_vit_b_01ec64.pth
```
 
### 3. Create outputs folder
 
```bash
mkdir outputs
```
 
### 4. Run the app
 
```bash
streamlit run app.py
```
 
Open browser at **http://localhost:8501**
 
---
 
## How It Works
 
1. Upload an image (JPG or PNG)
2. Select output mode from sidebar
3. Click **Run AI Segmentation**
4. YOLOv8 detects all objects in the image
5. SAM generates precise pixel-level masks for each detected object
6. Result is displayed with metrics and download option
---
 
## File Structure
 
```
Color Detect/
│
├── app.py              → Streamlit web app (main entry point)
├── main.py             → Standalone script using SAM only
├── models/
│   └── sam_vit_b_01ec64.pth  → SAM model checkpoint (download separately)
├── outputs/            → Saved result images
└── yolov8n.pt          → YOLOv8 model (auto-downloaded)
```
 
---
 
## Technologies
 
- **Python**
- **YOLOv8** — Ultralytics object detection
- **Meta SAM** — Segment Anything Model
- **Streamlit** — Web UI
- **OpenCV** — Image processing
- **PyTorch** — Deep learning backend"test" 
"test" 
