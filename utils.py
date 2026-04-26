# utils.py — Shared utilities for the Weapon Detection System

from __future__ import annotations

import base64
import json
import math
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import ALERT_CFG, CLASS_NAMES, RULE_CFG, WEAPON_CLASS_IDS
from logger import get_logger

log = get_logger("utils")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Detection:
    """Single object detection result."""
    class_id:   int
    class_name: str
    conf:       float
    bbox:       tuple[int, int, int, int]   # x1, y1, x2, y2
    track_id:   Optional[int] = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return float((x2 - x1) * (y2 - y1))

    def is_weapon(self) -> bool:
        return self.class_id in WEAPON_CLASS_IDS


@dataclass
class Alert:
    """Alert fired by the rule engine."""
    rule:           str
    weapon:         Detection
    target_person:  Optional[Detection]
    frame_idx:      int
    timestamp:      float = field(default_factory=time.time)
    snapshot_path:  Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "rule":         self.rule,
            "weapon":       self.weapon.class_name,
            "weapon_conf":  round(self.weapon.conf, 3),
            "weapon_bbox":  self.weapon.bbox,
            "person_bbox":  self.target_person.bbox if self.target_person else None,
            "frame":        self.frame_idx,
            "timestamp":    self.timestamp,
            "snapshot":     self.snapshot_path,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# BOUNDING-BOX MATH
# ═══════════════════════════════════════════════════════════════════════════════

def bbox_iou(a: tuple, b: tuple) -> float:
    """Intersection-over-Union for two (x1,y1,x2,y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union


def center_distance(a: Detection, b: Detection) -> float:
    """Euclidean distance between centres of two detections."""
    cx1, cy1 = a.center
    cx2, cy2 = b.center
    return math.hypot(cx2 - cx1, cy2 - cy1)


def frame_diagonal(frame: np.ndarray) -> float:
    h, w = frame.shape[:2]
    return math.hypot(w, h)


def weapon_pointing_angle(weapon: Detection, person: Detection) -> float:
    """
    Approximate angle (degrees) between the weapon bbox's long axis and the
    vector from weapon centre → person centre.
    Returns 0–90 (0 = perfectly pointing at person).
    """
    wx1, wy1, wx2, wy2 = weapon.bbox
    # weapon orientation vector (long axis)
    ww, wh = wx2 - wx1, wy2 - wy1
    if ww == 0 and wh == 0:
        return 90.0
    weapon_vec = np.array([ww, wh], dtype=float)

    # vector from weapon to person
    cx1, cy1 = weapon.center
    cx2, cy2 = person.center
    to_person = np.array([cx2 - cx1, cy2 - cy1], dtype=float)

    if np.linalg.norm(to_person) == 0:
        return 90.0

    cos_theta = np.dot(weapon_vec, to_person) / (
        np.linalg.norm(weapon_vec) * np.linalg.norm(to_person)
    )
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    angle = math.degrees(math.acos(abs(cos_theta)))
    return min(angle, 90.0 - angle) if angle > 45 else angle


# ═══════════════════════════════════════════════════════════════════════════════
# RULE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class RuleEngine:
    """
    Stateful rule engine that analyses per-frame detections and fires Alerts.
    Rules implemented:
      1. proximity     — weapon too close to a person
      2. pointing      — weapon oriented toward a person
      3. under_desk    — weapon partially occluded in lower half of frame
    """

    def __init__(self, frame_shape: tuple[int, int]):
        self.h, self.w = frame_shape[:2]
        self.diagonal = math.hypot(self.w, self.h)
        self._prox_thresh  = RULE_CFG["proximity_threshold_ratio"] * self.diagonal
        self._angle_thresh = RULE_CFG["pointing_angle_threshold"]
        self._occl_thresh  = RULE_CFG["occlusion_overlap_ratio"]
        self._temp_window  = RULE_CFG["temporal_window_frames"]
        self._alert_count  = RULE_CFG["alert_frame_count"]
        self._cooldown     = RULE_CFG["alert_cooldown_seconds"]

        # Temporal buffers: (weapon_track_id, person_track_id) → deque of booleans
        self._history: dict[tuple, deque] = {}
        # Last alert time per pair
        self._last_alert: dict[tuple, float] = {}

    def _pair_key(self, w: Detection, p: Optional[Detection]) -> tuple:
        return (w.track_id or id(w), p.track_id if p else -1)

    def _check_temporal(self, key: tuple, fired: bool) -> bool:
        buf = self._history.setdefault(key, deque(maxlen=self._temp_window))
        buf.append(fired)
        if sum(buf) >= self._alert_count:
            now = time.time()
            if now - self._last_alert.get(key, 0) >= self._cooldown:
                self._last_alert[key] = now
                return True
        return False

    def evaluate(
        self,
        frame_idx: int,
        detections: list[Detection],
    ) -> list[Alert]:
        alerts: list[Alert] = []
        weapons = [d for d in detections if d.is_weapon()]
        persons = [d for d in detections if not d.is_weapon()]

        for weapon in weapons:
            nearest_person = self._nearest_person(weapon, persons)

            # ── Rule 1: Proximity ────────────────────────────────────────────
            if nearest_person:
                dist = center_distance(weapon, nearest_person)
                fired = dist < self._prox_thresh
                key = ("proximity", *self._pair_key(weapon, nearest_person))
                if self._check_temporal(key, fired):
                    alerts.append(Alert("proximity_threat", weapon, nearest_person, frame_idx))

            # ── Rule 2: Pointing ─────────────────────────────────────────────
            if nearest_person:
                angle = weapon_pointing_angle(weapon, nearest_person)
                fired = angle < self._angle_thresh
                key = ("pointing", *self._pair_key(weapon, nearest_person))
                if self._check_temporal(key, fired):
                    alerts.append(Alert("weapon_pointing", weapon, nearest_person, frame_idx))

            # ── Rule 3: Under-desk occlusion (weapon in bottom 40 % of frame) ─
            _, wy1, _, wy2 = weapon.bbox
            desk_y = int(self.h * 0.60)
            overlap_h = max(0, wy2 - max(wy1, desk_y))
            weapon_h  = max(1, wy2 - wy1)
            fired = (overlap_h / weapon_h) >= self._occl_thresh
            key = ("under_desk", *self._pair_key(weapon, None))
            if self._check_temporal(key, fired):
                alerts.append(Alert("concealed_weapon", weapon, nearest_person, frame_idx))

        return alerts

    @staticmethod
    def _nearest_person(
        weapon: Detection, persons: list[Detection]
    ) -> Optional[Detection]:
        if not persons:
            return None
        return min(persons, key=lambda p: center_distance(weapon, p))


# ═══════════════════════════════════════════════════════════════════════════════
# DRAWING / ANNOTATION
# ═══════════════════════════════════════════════════════════════════════════════

# Colour palette per class (BGR)
_COLORS: dict[int, tuple] = {
    0: (180, 180, 180),   # person — grey
    1: (0,   0,   220),   # gun    — red
    2: (0,   140, 255),   # knife  — orange
    3: (0,   0,   180),   # rifle  — dark red
    4: (20,  20,  200),   # handgun— red variant
}
_ALERT_COLOR = (0, 0, 255)


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw bounding boxes and labels for all detections."""
    out = frame.copy()
    for det in detections:
        color = _COLORS.get(det.class_id, (200, 200, 200))
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.conf:.2f}"
        if det.track_id is not None:
            label = f"[{det.track_id}] {label}"
        _put_label(out, label, x1, y1, color)
    return out


def draw_alerts(frame: np.ndarray, alerts: list[Alert]) -> np.ndarray:
    """Highlight alert bounding boxes and overlay rule name."""
    out = frame
    for alert in alerts:
        x1, y1, x2, y2 = alert.weapon.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), _ALERT_COLOR, 3)
        cv2.putText(
            out, f"⚠ {alert.rule.upper()}", (x1, max(y1 - 10, 10)),
            cv2.FONT_HERSHEY_DUPLEX, 0.65, _ALERT_COLOR, 2,
        )
        if alert.target_person:
            px1, py1, px2, py2 = alert.target_person.bbox
            cv2.rectangle(out, (px1, py1), (px2, py2), _ALERT_COLOR, 2)
            # draw line from weapon to person
            wc = (int((x1+x2)/2), int((y1+y2)/2))
            pc = (int((px1+px2)/2), int((py1+py2)/2))
            cv2.line(out, wc, pc, _ALERT_COLOR, 2)
    return out


def draw_hud(
    frame: np.ndarray,
    fps: float,
    frame_idx: int,
    n_alerts: int,
) -> np.ndarray:
    """Draw HUD overlay (FPS, frame count, alert count)."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    cv2.putText(frame, f"FPS: {fps:5.1f}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 1)
    cv2.putText(frame, f"Frame: {frame_idx}", (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    alert_color = (0, 60, 255) if n_alerts else (0, 200, 80)
    cv2.putText(frame, f"Alerts: {n_alerts}", (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, alert_color, 1)
    return frame


def _put_label(
    img: np.ndarray, text: str, x: int, y: int, color: tuple
) -> None:
    """Draw a filled rectangle + text label above a bbox."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y0 = max(y - th - 6, th + 6)
    cv2.rectangle(img, (x, y0 - th - 4), (x + tw + 4, y0 + 2), color, -1)
    cv2.putText(img, text, (x + 2, y0), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
# ALERT OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

def save_alert_snapshot(frame: np.ndarray, alert: Alert) -> str:
    """Save annotated frame to disk and return path."""
    snap_dir = Path(ALERT_CFG["snapshot_dir"])
    snap_dir.mkdir(parents=True, exist_ok=True)
    ts = int(alert.timestamp * 1000)
    path = snap_dir / f"alert_{alert.rule}_{ts}.jpg"
    cv2.imwrite(str(path), frame)
    return str(path)


def post_alert_webhook(alert: Alert) -> None:
    """POST alert JSON to configured webhook URL (fire-and-forget)."""
    url = ALERT_CFG.get("webhook_url")
    if not url:
        return
    payload = json.dumps(alert.to_dict()).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=2):
            pass
    except Exception as exc:
        log.warning("Webhook failed: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════════
# MISC HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

class FPSCounter:
    """Rolling-window FPS estimator."""

    def __init__(self, window: int = 30):
        self._times: deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        self._times.append(time.perf_counter())
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


def xyxy_to_xywh(bbox: tuple) -> tuple:
    x1, y1, x2, y2 = bbox
    return x1, y1, x2 - x1, y2 - y1


def xywh_to_xyxy(bbox: tuple) -> tuple:
    x, y, w, h = bbox
    return x, y, x + w, y + h


def clamp_bbox(bbox: tuple, w: int, h: int) -> tuple:
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(x1, w)), max(0, min(y1, h)),
        max(0, min(x2, w)), max(0, min(y2, h)),
    )
