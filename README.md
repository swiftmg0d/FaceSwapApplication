# FaceSwap - A Face Swapping Application

<div style="text-align: center;">
    <img alt="FaceSwap Demo" src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdTZ3dWNkYnVsNnoyeTY0czd2bm1hcnVkanRkYjBtdnF2NHo3enkyOSZlcD12MV9pbnRlcm5naWZfYnlfaWQmY3Q9Zw/2eAnxaZ92Ueg0yeRxg/giphy.gif">
</div>

## Table of Contents

- [Description](#description)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Key Features](#key-features)

## Description

FaceSwap is an application designed for swapping faces in images using deep learning and image processing techniques. The application provides both **sequential** and **multithreading** processing for optimized performance. The user interface is built with **Tkinter** and includes a demo UI to visualize the face-swapping process.

The application leverages **OpenCV** and **Dlib** for face detection and alignment, ensuring precise and seamless face swaps. Additionally, it supports **parallel processing** with multithreading to handle multiple image pairs efficiently.

## Tech Stack

- **Programming Language**: Python
- **Libraries**: OpenCV, Dlib, NumPy, Tkinter
- **Parallel Processing**: ThreadPoolExecutor for multithreading
- **UI Framework**: Tkinter

## Installation

Follow these steps to set up the project locally:

### 1. Clone the repository

```bash
git clone https://github.com/swiftmg0d/FaceSwapApplication.git
```

### 2. Navigate to the project directory

```bash
cd FaceSwapApplication
```

### 3. Install dependencies

Ensure you have Python installed, then install the required libraries:

```bash
pip install opencv-python dlib numpy
```

### 4. Download the required Dlib model

The application requires the `shape_predictor_68_face_landmarks.dat` file for facial landmark detection. Download it from [Dlib's official website](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and place it in the `assets/` directory.

### 5. Run the application

To start the FaceSwap application, navigate to the **faceswap** directory:

```bash
cd demo/faceswap
```

To run the application, click on:

```bash
demo_application.exe
```

## Key Features

- **Face Detection**: Uses Dlib's pre-trained model for detecting and aligning faces.
- **Face Landmarking**: Extracts key facial points for seamless face blending.
- **Sequential & Multithreading**: Supports both single-threaded and multithreaded processing for efficient face swapping.
- **User Interface**: Built with Tkinter, offering an easy-to-use demo UI.
- **Parallel Processing**: Uses `ThreadPoolExecutor` to process multiple face swaps simultaneously.
- **Error Handling**: Robust exception handling for missing faces or processing errors.
- **High-Quality Output**: Optimized blending and color correction for realistic results.

### Parallel and Sequential Processing Code

The **parallel** and **sequential** processing code are located in:

```
source-code/faceswap/
```

- `single.py`: Contains the **sequential** face-swapping logic.
- `multi.py`: Contains the **multithreading** implementation for faster processing.

This application is ideal for fun projects, research, and image editing. Enjoy seamless and efficient face swapping with FaceSwap!

