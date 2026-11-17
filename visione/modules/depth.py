import contextlib
import os

import cv2
import numpy as np
from depth_anything_3.api import DepthAnything3

from config import DEVICE, DEPTH_MODEL_VARIANT, DEPTH_SCALE


class DepthEstimator:
    def __init__(self):
        if not os.path.exists(DEPTH_MODEL_VARIANT):
            raise FileNotFoundError(f"Depth model not found: {DEPTH_MODEL_VARIANT}")

        self.model = DepthAnything3.from_pretrained(DEPTH_MODEL_VARIANT, local_files_only=True)
        self.model.to(DEVICE).eval()

    def estimate(self, frame: np.ndarray) -> tuple:
        height, width = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull):
            prediction = self.model.inference([frame_rgb])

        if prediction.depth is None:
            raise RuntimeError("Depth model returned no depth map.")

        depth = prediction.depth[0]
        if depth.shape[:2] != (height, width):
            depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
        depth = depth * DEPTH_SCALE

        if prediction.conf is None:
            confidence = np.ones_like(depth)
        else:
            confidence = prediction.conf[0]
            if confidence.shape[:2] != (height, width):
                confidence = cv2.resize(confidence, (width, height), interpolation=cv2.INTER_LINEAR)

        return depth, confidence
