import numpy as np
import time
from config import (
    DISTANCE_WARN_NEAR, SPEED_ALERT_THRESHOLD,
    COOLDOWN_SPEED, COOLDOWN_GENERAL, MIN_TRACK_AGE,
    INPUT_SIZE, CENTER_ZONE_RATIO
)

class SafetyEngine:
    def __init__(self):
        self.speed_history = {}

        self.warn_history = {}
        self.active_query = None
        self.last_query_result = None

        self.last_cleanup_time = time.time()

    def reset_query(self):
        self.last_query_result = None

    def process(self, fused_detections: list, depth_map: np.ndarray, query: str or None) -> tuple:
        """
        Returns: (text, should_interrupt, is_critical_speed)
        """
        current_time = time.time()
        normalized_query = query.strip().lower() if query else None
        if normalized_query != self.active_query:
            self.active_query = normalized_query
            self.last_query_result = None

        if current_time - self.last_cleanup_time > 30.0:
            self.warn_history = {
                k: v for k, v in self.warn_history.items()
                if current_time - v['time'] < 60.0
            }
            self.speed_history = {
                uid: timestamp for uid, timestamp in self.speed_history.items()
                if current_time - timestamp < COOLDOWN_SPEED
            }
            self.last_cleanup_time = current_time

        for label, _, box, _, track_id, distance, _, velocity, direction, track_age in fused_detections:
            if track_age < MIN_TRACK_AGE: continue

            speed_ms = abs(velocity[2]) if velocity is not None else 0.0

            location = self._get_location_text(box)
            is_center_threat = (location == "ahead")

            if direction == "approaching" and speed_ms > SPEED_ALERT_THRESHOLD:
                if not is_center_threat and (distance is not None and distance > 1.0):
                    continue

                uid = track_id if track_id != -1 else label

                last_speed_time = self.speed_history.get(uid)
                if last_speed_time is not None and current_time - last_speed_time < COOLDOWN_SPEED:
                    continue

                self.speed_history[uid] = current_time

                guidance = self._get_guidance(box)
                msg = f"Warning! {label} approaching fast! {guidance}"
                return msg, True, True

        candidates = [
            det for det in fused_detections
            if det[9] >= MIN_TRACK_AGE and det[5] is not None
            and np.isfinite(det[5]) and 0 < det[5] < DISTANCE_WARN_NEAR
        ]
        for det in sorted(candidates, key=lambda item: item[5]):
            label, _, box, _, track_id, dist, _, _, _, _ = det
            uid = track_id if track_id != -1 else label
            last_data = self.warn_history.get(uid)
            if last_data is not None:
                time_diff = current_time - last_data['time']
                dist_diff = abs(dist - last_data['dist'])
                if time_diff <= COOLDOWN_GENERAL and dist_diff <= 0.8:
                    continue
            self.warn_history[uid] = {'time': current_time, 'dist': dist}
            return f"{label} {dist:.1f} meters. {self._get_guidance(box)}", True, False

        h, w = depth_map.shape

        y_start = h // 3
        y_end = int(h * 0.85)

        x_start, x_end = w // 3, 2 * w // 3

        center_zone = depth_map[y_start:y_end, x_start:x_end]
        known_mask = np.zeros_like(depth_map, dtype=bool)
        for det in fused_detections:
             if det[6] is not None: known_mask |= det[6]

        unknown_mask = (~known_mask)[y_start:y_end, x_start:x_end]
        danger_pixels = (center_zone < DISTANCE_WARN_NEAR) & (center_zone > 0.1) & unknown_mask

        if np.sum(danger_pixels) > (center_zone.size * 0.15):
            uid = "generic_obstacle"
            should_speak = False
            if uid not in self.warn_history:
                should_speak = True
            else:
                if current_time - self.warn_history[uid]['time'] > COOLDOWN_GENERAL:
                    should_speak = True

            if should_speak:
                self.warn_history[uid] = {'time': current_time, 'dist': 1.0}
                return "Obstacle ahead. Stop.", True, False

        if normalized_query:
            return self._process_query(normalized_query, fused_detections)

        return None, False, False

    def _get_guidance(self, box):
        return "Stop."

    def _get_location_text(self, box):
        x1, _, x2, _ = box
        center_x = (x1 + x2) / 2
        w = INPUT_SIZE[0]

        safe_left = w * (0.5 - CENTER_ZONE_RATIO/2)
        safe_right = w * (0.5 + CENTER_ZONE_RATIO/2)

        if center_x < safe_left: return "on your left"
        elif center_x > safe_right: return "on your right"
        else: return "ahead"

    def _process_query(self, query, fused_detections):
        target = query.lower()
        best_match = None
        min_dist = 999.0
        best_box = None

        for label, _, box, _, _, distance, _, _, _, _ in fused_detections:
            if (target in label.lower()) or (label.lower() in target):
                if distance is not None and np.isfinite(distance) and 0 < distance < min_dist:
                    min_dist = distance
                    best_match = label
                    best_box = box

        result_key = (target, best_match)
        if result_key == self.last_query_result:
            return None, False, False
        self.last_query_result = result_key
        if best_match:
            loc = self._get_location_text(best_box)
            return f"Found {best_match} {loc}, {min_dist:.1f} meters.", False, False

        return f"I don't see the {target} yet.", False, False
