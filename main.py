#!/usr/bin/env python3
# main.py — Weapon Detection System — Unified Entry Point
#
# Commands:
#   python main.py preprocess   [--raw-images DIR] [--raw-labels DIR]
#   python main.py train        [--data YAML] [--epochs N] [--resume CKPT]
#   python main.py validate     [--model PT] [--data YAML] [--split val|test]
#   python main.py export       [--model PT] [--format onnx|engine|tflite]
#   python main.py run          [--model PT] [--source 0|video.mp4|image.jpg]
#   python main.py demo                          # webcam demo, auto fallback to sample
#
# All heavy imports are deferred per sub-command so startup is fast.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from logger import get_logger, log_system_info

log = get_logger("main")


# ─────────────────────────────────────────────────────────────────────────────
# SUB-COMMAND HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

def cmd_preprocess(args: argparse.Namespace) -> None:
    from preprocessing import build_dataset
    yaml_path = build_dataset(
        raw_images_dir=args.raw_images,
        raw_labels_dir=args.raw_labels,
        out_dir=args.out_dir,
        augment=not args.no_augment,
    )
    log.info("Preprocessing complete. data.yaml → %s", yaml_path)


def cmd_train(args: argparse.Namespace) -> None:
    from training import train
    extra = {}
    for key in ("epochs", "batch", "imgsz", "device"):
        val = getattr(args, key, None)
        if val is not None:
            extra[key] = val
    best = train(
        data_yaml=args.data,
        resume=args.resume,
        extra_args=extra or None,
    )
    log.info("Training done. Best weights → %s", best)


def cmd_validate(args: argparse.Namespace) -> None:
    from training import validate
    metrics = validate(
        model_path=args.model,
        data_yaml=args.data,
        split=args.split,
    )
    print("\n── Validation Metrics ──────────────────")
    for k, v in metrics.items():
        print(f"  {k:<15}: {v:.4f}")
    print("────────────────────────────────────────\n")


def cmd_export(args: argparse.Namespace) -> None:
    from training import export_model
    out = export_model(
        model_path=args.model,
        fmt=args.format,
        half=args.half,
    )
    log.info("Export complete → %s", out)


def cmd_run(args: argparse.Namespace) -> None:
    from inference import load_model, run_image, run_video
    from config import TRAINED_MODEL_PATH, EXPORTS_DIR

    model_path = args.model or TRAINED_MODEL_PATH
    if not Path(str(model_path)).exists():
        log.error(
            "Model not found: %s\n"
            "Train first with:  python main.py train",
            model_path,
        )
        sys.exit(1)

    model  = load_model(model_path)
    source = args.source

    src_path = Path(str(source))
    if src_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"} or src_path.is_dir():
        run_image(model, src_path, display=not args.no_display)
    else:
        try:
            source = int(source)
        except (ValueError, TypeError):
            pass
        run_video(
            model,
            source,
            display=not args.no_display,
            save_output=args.save,
            output_path=args.output or EXPORTS_DIR / "output.avi",
        )


def cmd_demo(args: argparse.Namespace) -> None:
    """
    Quick smoke-test: try to open webcam; if not available, synthesise a
    test video with a bounding-box overlay and run inference on it.
    """
    import cv2
    from config import TRAINED_MODEL_PATH, EXPORTS_DIR
    from inference import load_model, run_video

    log.info("Demo mode — checking for webcam…")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.release()
        log.info("Webcam found. Launching live demo (press Q to quit).")
        model = load_model(TRAINED_MODEL_PATH)
        run_video(model, 0, display=True)
    else:
        log.warning("No webcam found. Generating synthetic test video…")
        _run_synthetic_demo()


def _run_synthetic_demo() -> None:
    """
    Create a 5-second synthetic video containing coloured rectangles
    (simulating a weapon near a person) and run the inference pipeline on it.
    """
    import cv2
    import numpy as np
    from config import EXPORTS_DIR

    out_path = EXPORTS_DIR / "demo_synthetic.avi"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(out_path), fourcc, 15, (640, 480))

    for i in range(75):   # 5 s @ 15 fps
        frame = np.full((480, 640, 3), 50, dtype=np.uint8)
        # simulate person (grey box)
        cv2.rectangle(frame, (200, 100), (320, 400), (160, 160, 160), -1)
        # simulate weapon (red box, moving closer)
        wx = 420 - i * 2
        cv2.rectangle(frame, (wx, 180), (wx + 60, 220), (0, 0, 200), -1)
        cv2.putText(frame, "SYNTHETIC DEMO", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        writer.write(frame)

    writer.release()
    log.info("Synthetic demo video saved → %s", out_path)
    log.info("Run:  python main.py run --source %s  (requires trained model)", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="main.py",
        description="Weapon Detection System — Edge AI Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py preprocess
  python main.py train --epochs 50
  python main.py validate --split test
  python main.py export --format onnx
  python main.py run --source 0
  python main.py run --source video.mp4 --save
  python main.py demo
        """,
    )
    root.add_argument("--verbose", action="store_true", help="Set log level to DEBUG")
    sub = root.add_subparsers(dest="command", required=True)

    # ── preprocess ────────────────────────────────────────────────────────────
    pp = sub.add_parser("preprocess", help="Build YOLO dataset from raw images+labels")
    pp.add_argument("--raw-images", type=Path, metavar="DIR")
    pp.add_argument("--raw-labels", type=Path, metavar="DIR")
    pp.add_argument("--out-dir",    type=Path, metavar="DIR")
    pp.add_argument("--no-augment", action="store_true",
                    help="Skip offline augmentation")

    # ── train ─────────────────────────────────────────────────────────────────
    tr = sub.add_parser("train", help="Fine-tune YOLOv8 on weapon dataset")
    tr.add_argument("--data",   type=Path, metavar="YAML")
    tr.add_argument("--resume", type=str,  metavar="CKPT", help="Resume from checkpoint")
    tr.add_argument("--epochs", type=int)
    tr.add_argument("--batch",  type=int)
    tr.add_argument("--imgsz",  type=int)
    tr.add_argument("--device", type=str)

    # ── validate ──────────────────────────────────────────────────────────────
    va = sub.add_parser("validate", help="Evaluate mAP / precision / recall")
    va.add_argument("--model", type=Path)
    va.add_argument("--data",  type=Path)
    va.add_argument("--split", type=str, default="val", choices=["val","test"])

    # ── export ────────────────────────────────────────────────────────────────
    ex = sub.add_parser("export", help="Export weights to ONNX/TensorRT/TFLite")
    ex.add_argument("--model",  type=Path)
    ex.add_argument("--format", type=str, default="onnx",
                    choices=["onnx","engine","tflite","coreml","ncnn"])
    ex.add_argument("--half",   action="store_true", help="FP16 export")

    # ── run ───────────────────────────────────────────────────────────────────
    ru = sub.add_parser("run", help="Real-time inference on camera/video/image")
    ru.add_argument("--model",      type=Path)
    ru.add_argument("--source",     default="0",
                    help="Camera index, video file, image file, or RTSP URL")
    ru.add_argument("--no-display", action="store_true")
    ru.add_argument("--save",       action="store_true",
                    help="Write annotated video to disk")
    ru.add_argument("--output",     type=Path)

    # ── demo ──────────────────────────────────────────────────────────────────
    sub.add_parser("demo", help="Quick smoke-test (webcam or synthetic video)")

    return root


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    if args.verbose:
        from logger import set_level
        set_level("DEBUG")

    log_system_info(log)

    dispatch = {
        "preprocess": cmd_preprocess,
        "train":      cmd_train,
        "validate":   cmd_validate,
        "export":     cmd_export,
        "run":        cmd_run,
        "demo":       cmd_demo,
    }
    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        handler(args)
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    except Exception as exc:
        log.exception("Fatal error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
