import numpy as np
import cv2
from collections import deque
from config import TRACK_BUFFER, VELOCITY_THRESHOLD, VELOCITY_SMOOTHING
from filterpy.kalman import KalmanFilter


class ObjectTrack:
    """Store measured history and missed frames for one object."""
    def __init__(self, kf, initial_age=1):
        self.kf = kf
        self.age = initial_age
        self.missed_frames = 0
        self.vel_history = deque(maxlen=VELOCITY_SMOOTHING)


class VelocityTracker:
    def __init__(self):
        self.tracks = {}

    def update(self, detections, depth, dt):
        updated_detections = []
        if not np.isfinite(dt) or dt <= 0.001:
            dt = 0.033
        if dt > 1.0:
            dt = 1.0

        for track in self.tracks.values():
            track.missed_frames += 1
            track.kf.F[0, 3] = dt
            track.kf.F[1, 4] = dt
            track.kf.F[2, 5] = dt
            track.kf.predict()

        valid_depth = np.isfinite(depth) & (depth > 0)
        for label, mask, box, conf, track_id in detections:
            valid_mask = None
            if mask is not None and track_id != -1 and mask.shape == depth.shape:
                valid_mask = mask.astype(bool) & valid_depth
            if valid_mask is None or not np.any(valid_mask):
                updated_detections.append((label, mask, box, conf, track_id, np.zeros(3), "stationary", 0))
                continue
            moments = cv2.moments(valid_mask.astype(np.uint8))
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']
            centroid_3d = np.array([cx, cy, np.median(depth[valid_mask])])

            if track_id not in self.tracks:
                kf = KalmanFilter(dim_x=6, dim_z=3)
                kf.F = np.eye(6)
                kf.H = np.array([[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0]])
                kf.P *= 10.0
                kf.R *= 0.1
                kf.Q *= 0.01
                kf.x[:3] = centroid_3d.reshape((3, 1))
                self.tracks[track_id] = ObjectTrack(kf)
            else:
                track = self.tracks[track_id]
                track.age += 1
                track.kf.update(centroid_3d)
            track = self.tracks[track_id]
            track.missed_frames = 0
            track.vel_history.append(float(track.kf.x[5, 0]))
            raw_velocity = track.kf.x[3:6].flatten()
            final_vz = float(np.mean(track.vel_history))
            final_velocity = np.array([raw_velocity[0], raw_velocity[1], final_vz])
            if abs(final_vz) < VELOCITY_THRESHOLD:
                direction = "stationary"
            elif final_vz < -VELOCITY_THRESHOLD:
                direction = "approaching"
            elif final_vz > VELOCITY_THRESHOLD:
                direction = "receding"
            else:
                direction = "moving"
            updated_detections.append((label, mask, box, conf, track_id, final_velocity, direction, track.age))

        self.tracks = {
            tid: track for tid, track in self.tracks.items()
            if track.missed_frames <= TRACK_BUFFER
        }
        return updated_detections
