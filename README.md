

Add a "flame" effect on each hand's index onto a video stream.

This script is just a quick hack, it's a bit of glue between [mediapipe](https://google.github.io/mediapipe/) for hand detection, [VidGear](https://abhitronix.github.io/vidgear/latest/) for fast frame fetching and [pyvirtualcam](https://github.com/letmaik/pyvirtualcam) for the virtual camera. Also a basic particle system to draw a flame of dubious artistic quality.

Should work on Linux/MacOS/Windows, through tested on Mac only.

## Setup

Install dependencies:
```sh
pip install pyvirtualcam opencv-python-headless vidgear numpy mediapipe
```

Then setup your camera (only need to do this once), by following: https://github.com/letmaik/pyvirtualcam#supported-virtual-cameras

## Usage

Start this script, then in services that use a webcam you can select OBS as a source.

Some services will overtake the webcam by default (example Google Meet), so this script will be stopped. Just restart it after switching the the OBS source.
