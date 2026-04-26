# inference.py — Real-Time Inference Engine for Weapon Detection
#
# Modes:
#   • live camera stream
#   • video file
#   • single image / directory of images
#
# Usage:
#   python inference.py --source 0                         # webcam
#   python inference.py --source path/to/video.mp4
#   python inference.py --source path/to/image.jpg --no-display

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Union

import cv2
import numpy as np

from config import (
    ALERT_CFG,
    CAMERA_SOURCE,
    DISPLAY_SCALE,
    EXPORTS_DIR,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    INFER_CFG,
    TARGET_FPS,
    TRAINED_MODEL_PATH,
)
from logger import get_logger, log_system_info
from utils import (
    Alert,
    Detection,
    FPSCounter,
    RuleEngine,
    draw_alerts,
    draw_detections,
    draw_hud,
    post_alert_webhook,
    save_alert_snapshot,
)

log = get_logger("inference")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path: Union[str, Path]):
    """Load YOLOv8 model (YOLO class auto-detects .pt / .onnx / .engine)."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics not installed. Run: pip install ultralytics")

    path = str(model_path)
    log.info("Loading model: %s", path)
    return YOLO(path)


# ─────────────────────────────────────────────────────────────────────────────
# RESULT → Detection CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

def _parse_results(results, frame_w: int, frame_h: int) -> list[Detection]:
    """
    Convert Ultralytics Results object → list[Detection].
    Works with detection models (and gracefully with segmentation / pose).
    """
    from config import CLASS_NAMES

    detections: list[Detection] = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id  = int(box.cls[0])
            conf    = float(box.conf[0])
            xyxy    = box.xyxy[0].cpu().numpy().astype(int).tolist()
            track_id = int(box.id[0]) if (box.id is not None) else None

            x1 = max(0, min(xyxy[0], frame_w - 1))
            y1 = max(0, min(xyxy[1], frame_h - 1))
            x2 = max(0, min(xyxy[2], frame_w - 1))
            y2 = max(0, min(xyxy[3], frame_h - 1))

            detections.append(Detection(
                class_id   = cls_id,
                class_name = CLASS_NAMES.get(cls_id, str(cls_id)),
                conf       = conf,
                bbox       = (x1, y1, x2, y2),
                track_id   = track_id,
            ))
    return detections


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-FRAME INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def infer_frame(model, frame: np.ndarray, track: bool = True) -> list[Detection]:
    """
    Run detection (+ optional tracking) on a single BGR frame.
    Returns list of Detection objects.
    """
    h, w = frame.shape[:2]
    if track:
        results = model.track(
            frame,
            conf     = INFER_CFG["conf"],
            iou      = INFER_CFG["iou"],
            imgsz    = INFER_CFG["imgsz"],
            device   = INFER_CFG["device"],
            half     = INFER_CFG["half"],
            max_det  = INFER_CFG["max_det"],
            verbose  = INFER_CFG["verbose"],
            persist  = True,   # maintain track IDs across frames
        )
    else:
        results = model.predict(
            frame,
            conf    = INFER_CFG["conf"],
            iou     = INFER_CFG["iou"],
            imgsz   = INFER_CFG["imgsz"],
            device  = INFER_CFG["device"],
            half    = INFER_CFG["half"],
            max_det = INFER_CFG["max_det"],
            verbose = INFER_CFG["verbose"],
        )
    return _parse_results(results, w, h)


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE MODE
# ─────────────────────────────────────────────────────────────────────────────

def run_image(
    model,
    source: Path,
    display: bool = True,
    save_dir: Path = EXPORTS_DIR / "images",
) -> None:
    """Run detection on a single image or a directory of images."""
    save_dir.mkdir(parents=True, exist_ok=True)
    paths = [source] if source.is_file() else sorted(source.glob("*"))
    paths = [p for p in paths if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"}]

    for img_path in paths:
        frame = cv2.imread(str(img_path))
        if frame is None:
            log.warning("Cannot read: %s", img_path)
            continue

        h, w = frame.shape[:2]
        detections = infer_frame(model, frame, track=False)

        # Dummy rule engine for static image
        engine = RuleEngine(frame.shape)
        alerts = engine.evaluate(0, detections)

        annotated = draw_detections(frame, detections)
        annotated = draw_alerts(annotated, alerts)

        out_path = save_dir / img_path.name
        cv2.imwrite(str(out_path), annotated)
        log.info("%s → %d detections, %d alerts → saved %s",
                 img_path.name, len(detections), len(alerts), out_path)

        if display:
            cv2.imshow("Weapon Detection", annotated)
            cv2.waitKey(0)

    if display:
        cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO / LIVE STREAM MODE
# ─────────────────────────────────────────────────────────────────────────────

def run_video(
    model,
    source: Union[str, int, Path],
    display: bool = True,
    save_output: bool = False,
    output_path: Path = EXPORTS_DIR / "output.avi",
) -> None:
    """
    Main real-time inference loop.

    Parameters
    ----------
    model        : Loaded YOLO model.
    source       : Camera index (int), video path (Path/str), or RTSP URL.
    display      : Show live annotated window.
    save_output  : Write annotated video to disk.
    output_path  : Destination .avi file.
    """
    cap = cv2.VideoCapture(str(source) if isinstance(source, Path) else source)
    if not cap.isOpened():
        raise IOError(f"Cannot open video source: {source}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    real_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer: cv2.VideoWriter | None = None
    if save_output:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(
            str(output_path), fourcc, min(TARGET_FPS, real_fps),
            (FRAME_WIDTH, FRAME_HEIGHT),
        )
        log.info("Saving output video → %s", output_path)

    fps_counter = FPSCounter(window=30)
    engine: RuleEngine | None = None   # initialised on first frame
    frame_idx   = 0
    skip_step   = max(1, int(real_fps / TARGET_FPS))
    total_alerts = 0

    log.info("Inference started | source=%s | skip=%d/%d frames",
             source, skip_step - 1, skip_step)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log.info("Stream ended or no more frames.")
                break

            frame_idx += 1
            if frame_idx % skip_step != 0:
                continue   # throttle to TARGET_FPS

            # Lazy initialise rule engine with actual frame shape
            if engine is None:
                engine = RuleEngine(frame.shape)

            # ── Detect ───────────────────────────────────────────────────────
            detections = infer_frame(model, frame, track=True)

            # ── Behaviour rules ──────────────────────────────────────────────
            alerts: list[Alert] = engine.evaluate(frame_idx, detections)

            # ── Alert side-effects ───────────────────────────────────────────
            if alerts:
                total_alerts += len(alerts)
                for alert in alerts:
                    log.warning("ALERT [frame %d] rule=%s  weapon=%s  conf=%.2f",
                                frame_idx, alert.rule,
                                alert.weapon.class_name, alert.weapon.conf)
                    if ALERT_CFG["save_snapshots"]:
                        snap_frame = draw_detections(frame.copy(), detections)
                        snap_frame = draw_alerts(snap_frame, alerts)
                        alert.snapshot_path = save_alert_snapshot(snap_frame, alert)
                    post_alert_webhook(alert)

            # ── Annotation ───────────────────────────────────────────────────
            vis = draw_detections(frame, detections)
            vis = draw_alerts(vis, alerts)
            fps = fps_counter.tick()
            vis = draw_hud(vis, fps, frame_idx, total_alerts)

            # ── Display ──────────────────────────────────────────────────────
            if display:
                scale = DISPLAY_SCALE
                if scale != 1.0:
                    h, w = vis.shape[:2]
                    vis = cv2.resize(vis, (int(w*scale), int(h*scale)))
                cv2.imshow("Weapon Detection — Press Q to quit", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    log.info("User quit.")
                    break

            if writer:
                out_frame = cv2.resize(vis, (FRAME_WIDTH, FRAME_HEIGHT))
                writer.write(out_frame)

    finally:
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        log.info("Session ended. Total alerts: %d  Frames processed: %d",
                 total_alerts, frame_idx // skip_step)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Weapon Detection — Real-Time Inference"
    )
    p.add_argument("--model",       type=Path, default=TRAINED_MODEL_PATH,
                   help="Path to .pt / .onnx / .engine weights")
    p.add_argument("--source",      default=str(CAMERA_SOURCE),
                   help="Camera index, video path, image path, or RTSP URL")
    p.add_argument("--no-display",  action="store_true",
                   help="Suppress OpenCV window (headless mode)")
    p.add_argument("--save",        action="store_true",
                   help="Write annotated output video to disk")
    p.add_argument("--output",      type=Path,
                   default=EXPORTS_DIR / "output.avi",
                   help="Output video path (used with --save)")
    return p.parse_args()


def main():
    args   = _parse_args()
    log_system_info(log)

    model  = load_model(args.model)
    source = args.source

    # Auto-detect image vs video/stream
    src_path = Path(source)
    if src_path.suffix.lower() in {".jpg",".jpeg",".png",".bmp"} or (
        src_path.is_dir()
    ):
        run_image(model, src_path, display=not args.no_display)
    else:
        # Camera index?
        try:
            source = int(source)
        except ValueError:
            pass   # keep as string (file path or RTSP URL)

        run_video(
            model,
            source,
            display=not args.no_display,
            save_output=args.save,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
