import contextlib
import os
import time

import cv2
import torch.multiprocessing as mp

os.environ["NUMEXPR_MAX_THREADS"] = "16"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from modules.detector import OpenVocabDetector
from modules.tracker import VelocityTracker
from modules.fusion import FusionEngine
from modules.safety_engine import SafetyEngine
from modules.tts import VoiceGuide
from modules.whisper_listener import QueryListener
from utils.visualizer import Visualizer
from utils.logger import Logger
from config import INPUT_SIZE, SHOW_VISUALIZATION, SAFETY_VOCABULARY, WAIT_KEY_DELAY, WINDOW_TITLE


def run_visione(use_realsense=False):
    with contextlib.ExitStack() as resources:
        resources.callback(cv2.destroyAllWindows)
        logger = Logger()
        resources.callback(logger.close)
        detector = OpenVocabDetector()

        if use_realsense:
            from realsense.depth import DepthEstimator
        else:
            from modules.depth import DepthEstimator
        depth_est = DepthEstimator()
        if use_realsense:
            resources.callback(depth_est.stop)
            depth_est.start()

        tracker = VelocityTracker()
        fusion = FusionEngine()
        safety = SafetyEngine()
        vis = Visualizer()
        guide = VoiceGuide()
        resources.callback(guide.stop)
        listener = QueryListener(enabled=True)
        resources.callback(listener.stop)
        listener.start()

        if not use_realsense:
            cap = cv2.VideoCapture(0)
            resources.callback(cap.release)
            if not cap.isOpened():
                raise RuntimeError("Could not open camera 0.")

        last_time = time.monotonic()
        active_query = None
        pending_speech = {}
        logger.log("Visione Online.")

        while True:
            if use_realsense:
                frame = depth_est.get_live_frame()
                if frame is None:
                    raise RuntimeError("RealSense returned no frame.")
            else:
                ret, frame = cap.read()
                if not ret:
                    break

            now = time.monotonic()
            dt = now - last_time
            last_time = now
            frame = cv2.resize(frame, INPUT_SIZE)

            query = listener.get_query()
            if query:
                active_query = query.strip()
                safety.reset_query()
                pending_speech.pop(0, None)
                logger.log(f"Query: {active_query}", "USER")

            prompts = SAFETY_VOCABULARY.copy()
            if active_query and active_query not in prompts:
                prompts.append(active_query)

            detections = detector.detect(frame, prompts)
            depth, _ = depth_est.estimate(frame)
            updated = tracker.update(detections, depth, dt)
            fused = fusion.fuse(updated, depth)
            text, should_interrupt, is_critical = safety.process(fused, depth, active_query)

            if text:
                priority = 2 if is_critical else int(should_interrupt)
                pending_speech[priority] = text
            if pending_speech:
                priority = max(pending_speech)
                message = pending_speech[priority]
                if guide.speak(message, interrupt=priority > 0, priority=priority):
                    logger.log(f"Speak: {message}", "URGENT" if priority == 2 else "INFO")
                    del pending_speech[priority]

            if SHOW_VISUALIZATION:
                combined = vis.draw_composite(frame, fused, depth)
                cv2.imshow(WINDOW_TITLE, combined)
                logger.save_frame(combined)

            if cv2.waitKey(WAIT_KEY_DELAY) & 0xFF == ord('q'):
                break


if __name__ == "__main__":
    mp.freeze_support()
    run_visione()
