# training.py — YOLOv8 Fine-Tuning Pipeline for Weapon Detection
#
# Usage:
#   python training.py                          # use defaults from config.py
#   python training.py --data data/processed/data.yaml --epochs 100
#   python training.py --resume models/weapon_run/weights/last.pt

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from config import (
    BASE_MODEL,
    EXPORT_FORMAT,
    MODELS_DIR,
    PROCESSED_DIR,
    TRAIN_CFG,
    TRAINED_MODEL_PATH,
)
from logger import get_logger, log_system_info
from preprocessing import build_dataset

log = get_logger("training")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_data_yaml(data_yaml: Optional[Path]) -> Path:
    """Resolve data.yaml path; auto-build dataset if missing."""
    if data_yaml and data_yaml.exists():
        return data_yaml
    default = PROCESSED_DIR / "data.yaml"
    if default.exists():
        log.info("Using existing data.yaml: %s", default)
        return default
    log.info("data.yaml not found — running preprocessing pipeline...")
    return build_dataset()


def _load_model(resume: Optional[str]):
    """Load a YOLO model for training (fresh or resumed)."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics not installed. Run: pip install ultralytics")

    if resume:
        log.info("Resuming from checkpoint: %s", resume)
        model = YOLO(resume)
    else:
        log.info("Starting from base model: %s", BASE_MODEL)
        model = YOLO(BASE_MODEL)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(
    data_yaml:  Optional[Path] = None,
    resume:     Optional[str]  = None,
    extra_args: Optional[dict] = None,
) -> Path:
    """
    Fine-tune YOLOv8 on the weapon-detection dataset.

    Parameters
    ----------
    data_yaml  : Path to data.yaml (auto-generated if None).
    resume     : Path to a checkpoint .pt file to resume training.
    extra_args : Dict of additional Ultralytics train() kwargs that
                 override config.TRAIN_CFG.

    Returns
    -------
    Path to the best weights (.pt).
    """
    log_system_info(log)

    data_path = _get_data_yaml(data_yaml)
    model     = _load_model(resume)

    # Merge config + CLI overrides
    train_kwargs = {**TRAIN_CFG, "data": str(data_path)}
    if extra_args:
        train_kwargs.update(extra_args)

    log.info("Training config:")
    for k, v in train_kwargs.items():
        log.info("  %-20s : %s", k, v)

    # ── Train ────────────────────────────────────────────────────────────────
    log.info("Starting training…")
    results = model.train(**train_kwargs)

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    log.info("Training complete. Best weights: %s", best_weights)

    # Copy best weights to the canonical path
    if best_weights.exists():
        import shutil
        shutil.copy(best_weights, TRAINED_MODEL_PATH)
        log.info("Best weights saved → %s", TRAINED_MODEL_PATH)

    return best_weights


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate(
    model_path: Optional[Path] = None,
    data_yaml:  Optional[Path] = None,
    split:      str            = "val",
) -> dict:
    """
    Run Ultralytics validation and return a metrics dict.

    Parameters
    ----------
    model_path : .pt file to evaluate (defaults to TRAINED_MODEL_PATH).
    data_yaml  : data.yaml to evaluate against.
    split      : "val" | "test"
    """
    from ultralytics import YOLO

    mp   = model_path or TRAINED_MODEL_PATH
    data = _get_data_yaml(data_yaml)

    log.info("Validating %s on split='%s'", mp, split)
    model   = YOLO(str(mp))
    metrics = model.val(data=str(data), split=split)

    log.info("mAP@50     : %.4f", metrics.box.map50)
    log.info("mAP@50-95  : %.4f", metrics.box.map)
    log.info("Precision  : %.4f", metrics.box.mp)
    log.info("Recall     : %.4f", metrics.box.mr)

    return {
        "map50":     metrics.box.map50,
        "map50_95":  metrics.box.map,
        "precision": metrics.box.mp,
        "recall":    metrics.box.mr,
    }


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_model(
    model_path:  Optional[Path] = None,
    fmt:         str            = EXPORT_FORMAT,
    imgsz:       int            = 640,
    half:        bool           = False,
    dynamic:     bool           = False,
) -> Path:
    """
    Export trained weights to deployment format (ONNX, TensorRT, TFLite…).

    Returns path to exported file.
    """
    from ultralytics import YOLO

    mp    = model_path or TRAINED_MODEL_PATH
    log.info("Exporting %s → format=%s", mp, fmt)
    model = YOLO(str(mp))
    model.export(format=fmt, imgsz=imgsz, half=half, dynamic=dynamic)

    # Ultralytics writes the export next to the .pt
    suffix_map = {
        "onnx":    ".onnx",
        "engine":  ".engine",
        "tflite":  "_float32.tflite",
        "coreml":  ".mlpackage",
    }
    suffix   = suffix_map.get(fmt, f".{fmt}")
    out_path = mp.with_suffix(suffix)
    log.info("Export complete → %s", out_path)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Weapon Detection — Training Pipeline"
    )
    sub = p.add_subparsers(dest="command")

    # train
    t = sub.add_parser("train", help="Fine-tune YOLOv8")
    t.add_argument("--data",    type=Path, help="Path to data.yaml")
    t.add_argument("--resume",  type=str,  help="Checkpoint .pt to resume")
    t.add_argument("--epochs",  type=int)
    t.add_argument("--batch",   type=int)
    t.add_argument("--imgsz",   type=int)
    t.add_argument("--device",  type=str)

    # val
    v = sub.add_parser("val", help="Validate a trained model")
    v.add_argument("--model", type=Path)
    v.add_argument("--data",  type=Path)
    v.add_argument("--split", type=str, default="val")

    # export
    e = sub.add_parser("export", help="Export model to ONNX / TensorRT etc.")
    e.add_argument("--model", type=Path)
    e.add_argument("--format", type=str, default=EXPORT_FORMAT)
    e.add_argument("--half",   action="store_true")

    return p.parse_args()


def main():
    args = _parse_args()

    if args.command == "train" or args.command is None:
        extra = {}
        for key in ("epochs", "batch", "imgsz", "device"):
            val = getattr(args, key, None)
            if val is not None:
                extra[key] = val
        train(
            data_yaml=getattr(args, "data", None),
            resume=getattr(args, "resume", None),
            extra_args=extra or None,
        )

    elif args.command == "val":
        validate(
            model_path=args.model,
            data_yaml=args.data,
            split=args.split,
        )

    elif args.command == "export":
        export_model(
            model_path=args.model,
            fmt=args.format,
            half=args.half,
        )


if __name__ == "__main__":
    main()
