# Real-Time Object Detection and Counting

An AI-powered computer vision application for real-time object detection and counting using YOLO, OpenCV, and Python.

## Project Overview

This project uses the YOLO (You Only Look Once) object detection algorithm along with OpenCV to detect and count multiple objects in real time from a webcam or video. It identifies different object classes and displays the total number of detected objects on the screen.

## Features

- Real-time object detection
- Object counting by class
- Live webcam support
- Video file support
- YOLOv3-based detection
- Built with Python and OpenCV

## Technologies Used

- Python
- OpenCV
- YOLOv3
- NumPy

## Installation

Install the required libraries:

```bash
pip install opencv-python numpy
```

## Usage

1. Download the YOLO model files (`yolov3.weights`, `yolov3.cfg`, and `coco.names`).
2. Place them in the project directory.
3. Run the project:

```bash
python main.py
```

## Future Enhancements

- Upgrade to YOLOv8
- Improve counting accuracy with object tracking
- Export detection results
- Develop a web-based interface

## License

This project is licensed under the MIT License.