# Project: gametime

## Goal:
 - Offer a simple UI to allow users to create a virtual camera from multiple physical camera devices.
 - Create a a smart algorithm that is able to do face detection and switch the virtual camera to the correct camera, depending on which direction the user is looking.

## main.py

### Description
- Uses Python and OpenCV to display video from multiple webcams using threads.
- Each webcam feed appears in a separate window.
- Allows the user to specify the number of webcams and exits on 'Q' or 'Esc' key press.

### Usage
1. Run `main.py`.
2. Enter the number of connected webcams.
3. Each webcam's feed will display in its window.
4. Press 'Q' or 'Esc' to exit and stop all feeds.

## main.cpp

### Description
- C++ program using OpenCV to display video from two cameras in separate windows.
- Continuously captures and displays frames until exited with 'Esc'.

### Usage
1. Compile and run `main.cpp` with a C++ compiler.
2. Two windows will show the video feeds from the connected cameras.
3. Press 'Esc' to exit and close all windows.

## Directory Structure
gametime/
│
├── main.py
└── main.cpp

markdown
Copy code

### Dependencies
- Python 3.x
- OpenCV (Python and C++)

### Notes
- Ensure OpenCV and necessary libraries are installed.
- Adjust camera IDs or handle access errors based on your system setup.

### License
- Provided under an open-source license (if applicable).

### Author
- Richard Christopher 2023