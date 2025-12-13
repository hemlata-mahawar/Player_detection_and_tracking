## 🏏 Cricket Player Detection & Tracking Overview

This project implements a real-time cricket player detection and tracking system. It detects players from video frames and assigns persistent tracking IDs using modern deep learning and tracking algorithms.
In addition to core detection and tracking, the system includes  advanced analytical features such as player trajectory visualization, heatmap generation, and homography-based top-view transformation. These optional modules provide deeper insights into player movement patterns and spatial positioning, enhancing the overall analytical capabilities of the system.

## Tech Stack

- **YOLO12x** – Player Detection
- **BoT-SORT** – Multi-Object Tracking
- **GPU (CUDA)** – Accelerated Inference
- **OpenCV** – Video Processing & Visualization
- **NumPy & Pandas** – Data Handling & Analysis

## Features

- **Player Detection**: YOLO12l (YOLO-World v2 Large) for accurate player detection
- **Multi-Object Tracking**: BoT-SORT for consistent ID assignment across frames
- **Masked Detection**: Focus on player regions using masks
- **Trajectory Analysis**: Movement paths, heatmaps, and statistics
- **Top-View Projection**: Homography-based transformation to bird's-eye view
- **Visualization**: Multiple output formats with player IDs and trajectories

## 1. Clone the repository
```bash
git clone https://github.com/23f1001797/cricket_player_detection_and_top_view_implementation.git
cd cricket_player_detection_and_top_view_implementation
```
## Installation

1. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate 
   # On Windows use `venv\Scripts\activate`
   ```
2. **Install dependencies**:
   ```bash
    pip install -r requirements.txt
    ```

3. **how to run the code**:
   ```bash
    python run_advanced.py --video data/v3.mp4 --output output --ground data/ground.png --top-view data/top_view.png
    --show-mask True
   ```
   If you want you can run '''python run_advanced.py ''' only it will take default arguments.

   - --video: Path to input video file 
   - --output: Path to Directory to save output files 
   - --ground: Path to ground plane image for homography 
   - --top-view: Path to top-view reference image 
   - --show-mask: Whether to display the detection mask (True/False)

## Output
for an input video named {video_name}.mp4, the outputs will be saved in the output/{video_name}_output/ directory
everything will be saved in the following format:
```css
output/{video_name}_output/{video_name}_tracked.mp4 is the output video file with bounding boxes and tracking IDs.

output/{video_name}_output/top_view.mp4 is the bird's-eye view video showing player movements from a top-down perspective.
```

The system generates:
- {video_name}_tracked.mp4: A video file with bounding boxes and tracking IDs
- {video_name}_top_view.mp4: Top-view transformed video
- heatmap.png: Heatmap image of player movements
- enhanced_trajectories.mp4: Video with enhanced trajectory visualization
- trajectories.png: Visualized player trajectories
- tracking_data.json: JSON file with tracking metadata
- full_report.json: Comprehensive report with player statistics

All these outputs gets saved in the output directory and provide comprehensive insights into player detection, tracking, and movement analysis.

## folder structure

``` css
    cricket_player_detection/
    │
    ├── configs
    │   ├── botsort.yaml
    │   └── yolov12_config.yaml
    ├── data
    │   ├── ground.png
    │   ├── right_half.png
    │   ├── short.mp4
    │   ├── top_view.png
    │   └── v3.mp4
    ├── output
    │   ├── v1_output
    │   │   ├── v1_tracked.mp4
    │   │   ├── enhanced_trajectories.mp4
    │   │   ├── full_report.json
    │   │   ├── heatmap.png
    │   │   ├── top_view.mp4
    │   │   ├── tracking_data.json
    │   │   └── trajectories.png
    ├── README.md
    ├── run_advanced.py
    ├── src
    │   ├── detect_and_track.py
    │   ├── mask_select.py
    │   ├── homography_utils.py
    │   ├── trajectory_analyzer.py
    │   └── visualization.py
    └── yolo12x.pt
```