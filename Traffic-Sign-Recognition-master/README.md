# Real-Time Traffic Sign Detection and Recognition (TSDR)

[![Python](https://img.shields.io/badge/Python-3.7+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)

An intelligent, real-time computer vision system for detecting and recognizing road traffic signs from live video feeds. Built with OpenCV and Python, this project implements classical computer vision techniques and unsupervised machine learning without requiring heavy deep learning hardware, achieving high-frame-rate performance on standard edge CPUs.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture & Pipelines](#-system-architecture--pipelines)
- [Algorithms & Mathematical Foundations](#-algorithms--mathematical-foundations)
- [Supported Traffic Signs](#-supported-traffic-signs)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the System](#running-the-system)
- [Controls](#-controls)
- [Roadmap & Future Enhancements](#-roadmap--future-enhancements)
- [License](#-license)

---

## 📖 Overview

Traffic Sign Detection and Recognition (TSDR) is a core component of Advanced Driver Assistance Systems (ADAS) and Autonomous Vehicles (AVs). This repository demonstrates two distinct, complementary computer vision pipelines designed for real-time video processing:

1. **Circular & Dominant Color Pipeline (`road-sign-recognition.py`)**: Identifies circular signs (e.g., STOP, mandatory directional indicators) using Circular Hough Transform, K-Means color quantization, and spatial zone intensity analysis.
2. **Perspective-Rectified Blue Directional Sign Pipeline (`traffic sign recognition.py`)**: Isolates blue directional signs via HSV color segmentation and morphological filtering, corrects camera perspective skew via a 4-point perspective transform (homography), and classifies arrows using spatial grid density vectorization.

---

## ⚡ Key Features

- **Real-Time Video Capture**: Operates directly on live webcam feeds or recorded video streams.
- **Lighting-Invariant Color Filtering**: Leverages the HSV (Hue-Saturation-Value) color space to decouple illumination from chromatic information.
- **Planar Homography / Perspective Correction**: Warps tilted or skewed signs captured at an angle into standardized frontal views for invariant analysis.
- **Lightweight & GPU-Free**: Runs at 60+ FPS on edge CPUs with minimal memory overhead, zero neural network inference latency, and zero required training datasets.
- **Dual Recognition Schemes**: Employs both unsupervised clustering (K-Means) and spatial sub-block binary hash matching.

---

## 🛠 System Architecture & Pipelines

### Pipeline 1: Circle Detection & Dominant Color Analysis
```
Live Camera Stream ──► Grayscale ──► Median Filter (37x37) ──► Circular Hough Transform
                                                                      │
┌────────────────── Classification ◄── Spatial Zoning ◄── K-Means ROI Centroid
│ (STOP, LEFT, RIGHT, FORWARD, FORWARD & LEFT, FORWARD & RIGHT)
```

### Pipeline 2: HSV Segmentation & Perspective Rectification
```
Live Camera Stream ──► HSV Conversion ──► In-Range Blue Masking
                                                  │
┌──────────────── Morphological Opening & Closing (3x3 Kernel)
▼
Contour Extraction ──► MinAreaRotatedRect ──► 4-Point Homography Warp
                                                      │
┌────────────────── Classification ◄── 4-Bit Hash Lookup ◄── Sub-Block Density Analysis
│ (Turn Left, Turn Right, Move Straight, Turn Back)
```

---

## 🔬 Algorithms & Mathematical Foundations

### 1. Median Filtering
- **Function**: `cv2.medianBlur`
- **Purpose**: A non-linear digital filter that removes impulse ("salt-and-pepper") noise from input frames while preserving edge boundaries, avoiding spurious edges in Hough accumulator voting.

### 2. Circular Hough Transform (CHT)
- **Function**: `cv2.HoughCircles` (`cv2.HOUGH_GRADIENT`)
- **Purpose**: Detects circular geometry by combining Sobel gradient computations with accumulator voting in parameterized space $(x_c, y_c, r)$.

### 3. HSV Color Space Segmentation
- **Function**: `cv2.cvtColor`, `cv2.inRange`
- **Purpose**: Decouples luminance (Value channel) from chromaticity (Hue and Saturation), allowing robust color extraction across sunny, overcast, or shadowed conditions.

### 4. Mathematical Morphology
- **Function**: `cv2.morphologyEx`
- **Opening**: Erosion followed by dilation ($\mathbf{A} \circ \mathbf{B}$) to discard small background noise artifacts.
- **Closing**: Dilation followed by erosion ($\mathbf{A} \bullet \mathbf{B}$) to seal small holes and bridge disjoint segments within the detected sign mask.

### 5. Planar Perspective Rectification (Homography)
- **Function**: `four_point_transform` (`cv2.getPerspectiveTransform` + `cv2.warpPerspective`)
- **Purpose**: Computes a $3 \times 3$ perspective transformation matrix from the 4 corner points of the rotated bounding box to an upright rectangular plane, eliminating camera angle distortion.

### 6. K-Means Clustering (Color Quantization)
- **Function**: `cv2.kmeans`
- **Purpose**: Groups pixel colors into $K$ clusters via Euclidean distance minimization to isolate the dominant color centroid without hardcoded per-pixel thresholding.

### 7. Spatial Grid Density Vectorization
- **Methodology**: Subdivides the rectified sign into 4 distinct regions (`leftBlock`, `centerBlock`, `rightBlock`, `topBlock`). Each block's normalized active pixel density $\frac{\sum \text{pixels}}{\text{Area}}$ is thresholded to form a 4-bit signature vector `(left, center, right, top)` matched against a hash lookup table.

---

## 🚦 Supported Traffic Signs

| Sign Type | Detection Method | Classification Class |
| :--- | :--- | :--- |
| **Stop Sign** | Hough Circle + Dominant Red | `STOP` |
| **Turn Left** | Perspective Warp + Binary Grid / Hough Zoning | `Turn Left` / `LEFT` |
| **Turn Right** | Perspective Warp + Binary Grid / Hough Zoning | `Turn Right` / `RIGHT` |
| **Move Straight**| Perspective Warp + Binary Grid / Hough Zoning | `Move Straight` / `FORWARD` |
| **Turn Back / U-Turn** | Perspective Warp + Binary Grid | `Turn Back` |
| **Forward & Left** | Hough Circle + Zoning Intensity | `FORWARD AND LEFT` |
| **Forward & Right** | Hough Circle + Zoning Intensity | `FORWARD AND RIGHT` |

---

## 📁 Project Structure

```
Traffic-Sign-Recognition-master/
├── Signs/                         # Test sample images captured from real-world scenes
│   ├── IMG_4186.JPG
│   └── ...
├── Traffic Signs/                 # Reference sign icons and templates
│   ├── moveStraight.png
│   ├── turnBack.png
│   ├── turnLeft.png
│   ├── turnRight.png
│   └── No entry.png
├── road-sign-recognition.py       # Pipeline 1: Circle detection, K-Means & zoning
├── traffic sign recognition.py    # Pipeline 2: HSV segmentation, homography & grid vectorization
└── README.md                      # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.7 or higher
- A connected USB or built-in webcam

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vishali1435/Traffic-sign-detection-and-recognition.git
   cd Traffic-sign-detection-and-recognition/Traffic-Sign-Recognition-master
   ```

2. **Install required dependencies:**
   ```bash
   pip install opencv-python numpy imutils scipy
   ```

### Running the System

- **To run the Perspective Rectification & Directional Arrow Recognition:**
  ```bash
  python "traffic sign recognition.py"
  ```

- **To run the Circular Shape & Dominant Color Recognition:**
  ```bash
  python "road-sign-recognition.py"
  ```

---

## ⌨️ Controls

- **`traffic sign recognition.py`**: Press <kbd>q</kbd> while focusing on the camera window to close the application.
- **`road-sign-recognition.py`**: Click anywhere inside the camera window with the left mouse button to stop and exit.

---

## 🔮 Roadmap & Future Enhancements

- [ ] **CNN / Deep Learning Integration**: Incorporate lightweight MobileNetV3 or YOLOv8-Nano models trained on the German Traffic Sign Recognition Benchmark (GTSRB) for multi-class classification.
- [ ] **Adaptive Illumination Compensation**: Implement CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve reliability under direct glare or nighttime driving.
- [ ] **Temporal Consistency Tracking**: Integrate a Kalman Filter or centroid tracker to stabilize detections across consecutive frames.
- [ ] **Speed Limit Sign OCR**: Add Tesseract OCR / CRNN model to read numerical speed limits dynamically.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) - feel free to use, modify, and distribute for educational and commercial purposes.