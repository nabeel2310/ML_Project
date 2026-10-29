"""Dataset manifest checks; deliberately refuses unknown/incomplete class mappings."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "config" / "crop_classes.json"
REQUIRED_COLUMNS = {"image_path", "crop", "label", "symptom_text", "source", "group_id", "split"}
ALLOWED_SPLITS = {"train", "validation", "test"}


class ManifestError(ValueError):
    """Raised when a dataset manifest is unsafe or inconsistent."""


def load_class_manifest(path: Path = MANIFEST_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_class_manifest(manifest: dict) -> None:
    if manifest.get("status") != "complete":
        raise ManifestError("Class manifest is incomplete. Confirm and enter all 31 approved classes before training.")
    classes = manifest.get("classes", [])
    if len(classes) != manifest.get("expected_total_classes", 31):
        raise ManifestError("Class count does not match expected_total_classes.")
    ids = [c.get("id") for c in classes]
    keys = [(c.get("crop"), c.get("label")) for c in classes]
    allowed_crops = set(manifest.get("crops", []))
    if len(set(ids)) != len(ids) or len(set(keys)) != len(keys):
        raise ManifestError("Class ids and (crop, label) pairs must be unique.")
    for item in classes:
        if not item.get("id") or not item.get("label") or item.get("crop") not in allowed_crops:
            raise ManifestError(f"Invalid class entry: {item!r}")


def validate_dataset_manifest(csv_path: Path, *, require_files: bool = True) -> pd.DataFrame:
    frame = pd.read_csv(csv_path, keep_default_na=False)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ManifestError(f"Missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ManifestError("Dataset manifest is empty.")
    if (frame["image_path"].str.strip() == "").any():
        raise ManifestError("image_path contains empty values.")
    if (frame["label"].str.strip() == "").any() or (frame["crop"].str.strip() == "").any():
        raise ManifestError("crop and label must be non-empty.")
    if not set(frame["split"]) <= ALLOWED_SPLITS:
        raise ManifestError(f"split must be one of {sorted(ALLOWED_SPLITS)}; blank means not yet assigned.")
    if require_files:
        absent = [p for p in frame["image_path"] if not Path(p).is_file()]
        if absent:
            raise ManifestError(f"{len(absent)} image files are missing; first missing path: {absent[0]}")
    # Prevent the same group (e.g., same plant/plot) from crossing dataset splits.
    grouped = frame[frame["group_id"].str.strip() != ""]
    for group_id, rows in grouped.groupby("group_id"):
        splits = set(rows["split"]) - {""}
        if len(splits) > 1:
            raise ManifestError(f"group_id {group_id!r} leaks across splits: {sorted(splits)}")
    # A known source-level group is the preferred split unit. Exact image-path duplicates
    # must never cross splits either.
    for image_path, rows in frame.groupby("image_path"):
        splits = set(rows["split"]) - {""}
        if len(splits) > 1:
            raise ManifestError(f"image_path {image_path!r} leaks across splits.")
    return frame
