Add a "flame" effect on each hand's index onto a video stream.

https://user-images.githubusercontent.com/3440771/141673130-94892582-acac-4086-9cbf-1013e48b67c5.mov


This script is just a quick hack, it's a bit of glue between [mediapipe](https://google.github.io/mediapipe/) for hand detection, [VidGear](https://abhitronix.github.io/vidgear/latest/) for fast frame fetching and [pyvirtualcam](https://github.com/letmaik/pyvirtualcam) for the virtual camera. Also a basic particle system to draw a 🔥 (of dubious artistic quality).

Should work on Linux/MacOS/Windows, through tested on Mac only.

## Setup

Install dependencies:
```sh
python3 -m pip install pyvirtualcam opencv-python-headless vidgear numpy mediapipe
```

Then setup your camera (only need to do this once), by following: https://github.com/letmaik/pyvirtualcam#supported-virtual-cameras

## Usage

* `python3 demo.py`
* Select "OBS Virtual Camera" as a video source

Some services will overtake the webcam by default (example Google Meet), so this script will be stopped. If that happen, just restart it after switching to the OBS source.
