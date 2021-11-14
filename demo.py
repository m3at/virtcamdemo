#!/usr/bin/env python3

import sys
import time
import logging
import argparse

import cv2
import pyvirtualcam
import numpy as np
import mediapipe as mp
from vidgear.gears import CamGear

logger = logging.getLogger("base")

# Hands detection
mp_hands = mp.solutions.hands

# Camera settings
camgear_options = {
    "CAP_PROP_FRAME_WIDTH": 640,
    "CAP_PROP_FRAME_HEIGHT": 480,
    "CAP_PROP_FPS": 30,
}

rng = np.random.default_rng()

# Properties for the particle system, chosen to look like a flame, if you squint
# A bunch of globals because meh
P_SIZE = 0.08
P_SIZE_STD = abs(P_SIZE) / 3
VEL_Y = -0.012
VEL_Y_STD = abs(VEL_Y) / 5
VEL_X_STD = abs(VEL_Y) / 20
AGE_MAX = 35
AGE_SMALLEST = int(0.9 * AGE_MAX)

# Flame color over it's lifetime as RGB
table_colors = np.array(
    [
        [255, 255, 255],
        [254, 254, 201],
        [223, 159, 56],
        [201, 104, 39],
        [123, 42, 20],
        [50, 18, 18],
        [23, 14, 12],
    ],
    dtype=np.uint8,
)
MAX_COLOR = table_colors.shape[0] - 1


class ParticleSystem:
    def __init__(self, n=1_000):
        self.N = n

        # Keep track of the particles, as a big matrix to be faster
        self.particles_pos = np.zeros((n, 2), dtype=float)
        self.particles_vel = np.zeros((n, 2))
        self.particles_siz = np.full(n, P_SIZE, dtype=float)
        self.particles_age = np.arange(n, dtype=float) % AGE_MAX

    def updater(self, x, y):
        """Update particles state for the current frame."""

        # Age by one step and get expired ones
        p_age = self.particles_age
        p_age -= 1
        self.particles_age += rng.normal(0.0, scale=0.5, size=(self.N))
        mask = p_age < 0
        p_age %= 30

        # Initialize new ones
        m = np.sum(mask)
        new_pos = np.stack(
            (
                rng.normal(0.0, scale=0.015, size=(m)) + x,
                rng.normal(-0.01, scale=0.01, size=(m)) + y,
            )
        ).T
        new_vel = np.stack(
            (
                rng.normal(0.0, scale=VEL_X_STD, size=(m)),
                rng.normal(VEL_Y, scale=VEL_Y_STD, size=(m)),
            )
        ).T
        new_size = rng.normal(P_SIZE, scale=P_SIZE_STD, size=(m))

        # Update everything
        self.particles_pos[mask] = new_pos
        self.particles_vel[mask] = new_vel
        self.particles_siz[mask] = new_size.clip(0.01)
        self.particles_pos[~mask] = self.particles_vel[~mask] + self.particles_pos[~mask]

    def __call__(self, frame, x, y):
        self.updater(x, y)
        h, w, _ = frame.shape

        # Particles get smaller as they age, see where they stand
        evol = self.particles_age / AGE_SMALLEST

        # rage = radius + age. Funny (!?).
        rage = evol * w * P_SIZE
        rage *= self.particles_siz
        rage = rage.round().astype(int)

        # Colors
        r_evol = np.exp(evol * -1.5)
        colors = table_colors[(r_evol * MAX_COLOR).astype(int)].tolist()

        # Draw each particle as a circle
        for (a, b), r, c in zip(self.particles_pos, rage, colors):
            cv2.circle(
                frame,
                (int(a * w), int(b * h)),
                r,
                color=c,
                thickness=-1,
            )


def main() -> None:
    # Setup the virtual webcam
    ctx = pyvirtualcam.Camera(
        width=camgear_options["CAP_PROP_FRAME_WIDTH"],
        height=camgear_options["CAP_PROP_FRAME_HEIGHT"],
        fps=30,
    )

    with ctx as cam:
        # Open webcam stream
        # It fails sometimes, so just retry
        time.sleep(1.2)
        for i in range(3):
            try:
                stream = CamGear(source=1, logging=False, colorspace="COLOR_BGR2RGB", **camgear_options).start()
                break
            except RuntimeError:
                logger.debug(f"Retrying for camera, {i + 1}/3")
                time.sleep(0.5)
        else:
            logger.error("Couldn't initialize camera")
            sys.exit("nop")

        logger.info(f"Using virtual camera: {cam.device}")

        # Hand detector, the fastest one will do
        hand_detection = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            model_complexity=0,
        )

        # Setup one particle system per hand
        party_left = ParticleSystem()
        party_right = ParticleSystem()
        party = [party_left, party_right]

        while True:
            # Each frame is a (height, width, RGB) uint8 array
            in_frame = stream.read()

            if in_frame is None:
                logger.error("Invalid input frame")
                break

            # Detect hands keypoints
            in_frame.flags.writeable = False
            hands = hand_detection.process(in_frame)
            in_frame.flags.writeable = True

            # Draw if hands are detected
            if hands.multi_hand_landmarks:
                for (hand_landmarks, side) in zip(hands.multi_hand_landmarks, hands.multi_handedness):
                    index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    party[side.classification[0].index](in_frame, index.x, index.y)

            cam.send(in_frame)
            cam.sleep_until_next_frame()

    logger.info("Quit")


if __name__ == "__main__":
    # Get system arguments
    parser = argparse.ArgumentParser(
        description="Start a virtual webcam and add fire effect to index fingers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="DEBUG",
        choices=["debug", "info", "warning", "error"],
        help="Logging level",
    )
    args = vars(parser.parse_args())
    log_level = getattr(logging, args.pop("log_level").upper())

    # Setup logging
    logger.setLevel(log_level)
    ch = logging.StreamHandler()
    logger.propagate = False
    ch.setLevel(log_level)
    ch.setFormatter(logging.Formatter("{asctime} {levelname}| {message}", datefmt="%H:%M:%S", style="{"))
    logger.addHandler(ch)

    # Add colors
    _levels = [[226, "DEBUG"], [148, "INFO"], [208, "WARNING"], [197, "ERROR"]]
    for color, lvl in _levels:
        _l = getattr(logging, lvl)
        logging.addLevelName(_l, "\x1b[38;5;{}m{:<7}\x1b[0m".format(color, logging.getLevelName(_l)))

    try:
        main()
    except KeyboardInterrupt:
        logger.debug("KeyboardInterrupt")
