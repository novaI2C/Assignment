
# Computer Vision: Automated Sticker Placement (Brown Cardboard)

## Overview
This script detects a **brown cardboard box** on a conveyor-like background, estimates its orientation in the image plane, and computes the **(x, y)** pixel coordinates for centered sticker placement. The pipeline is classical CV (no ML): HSV segmentation, morphological cleanup, contour filtering, and rotated-rectangle fitting.

## Repository structure
```
computer_vision/
├── sticker_placement.py # final pipeline script
├── images/ # sample input images (e.g., tiltedbox.png)
├── outputs/ # example outputs (mask.png, final_output.png)
└── videos/ # demo recordings
```

## Dependencies
- `Python 3.8+`
- `opencv-python`
- `numpy`
- `matplotlib`

Install with:
```bash
pip install opencv-python numpy matplotlib
```
# How to run
```bash
python sticker_placement.py
```

The script opens a sequence of visualization windows (Matplotlib) showing each step:

Original image

HSV saturation channel

Raw brown mask (after cv2.inRange)

Cleaned mask (morphological closing/opening)

Selected box contour (after shape filtering)

Final output (box outline + oriented sticker + center point)

The script also prints assignment-ready numeric results:

```
Box center (px): (cx, cy)
Sticker center (px): (cx, cy)
Orientation angle (deg): θ`
```

# HSV calibration

HSV thresholds were empirically tuned using an interactive trackbar tool to isolate brown cardboard under the test lighting.

Final calibrated range used in the script:

Hue: 13 – 35

Saturation: >= 80

Value: >= 90

This strict brown range rejects conveyor background and non-target objects.

# Algorithm summary

Gaussian blur → convert to HSV.

HSV thresholding (strict brown) → binary mask.

Morphological closing then opening to remove holes/noise.

Contour extraction (cv2.findContours) with RETR_TREE.

Shape-based filtering:

Reject contours with area < 3000 px (noise).

Reject contours > 70% image area (background).

Accept contours with aspect ratio in range (1.2, 3.5).

Fit minAreaRect to selected contour, normalize angle.

Place sticker rectangle centered at box center and rotated by the same angle.

Display intermediate visualizations and print (x, y, θ).

# Outputs

Final annotated image (box contour, sticker region, center).

Numeric outputs for robotic actuation: center coordinates and orientation.

# Limitations & notes

Designed for brown cardboard boxes; other colors require re-calibration.

Assumes a single box in the scene.

Outputs coordinates in image pixels. For robot mm coordinates, camera calibration + homography/pinhole model are required.

# Where to find results

Saved example outputs can be found in cv/outputs/.

Screen recordings (if created) are stored in cv/videos/.

# Contact / reproduction

Use the provided sample images to reproduce the visual results.

For different lighting or camera setups, run the included interactive HSV tuner to obtain new thresholds before running the pipeline.
