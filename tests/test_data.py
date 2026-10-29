import pandas as pd
import pytest

from src.crop_disease.data import ManifestError, validate_class_manifest, validate_dataset_manifest


def test_incomplete_class_manifest_blocks_training():
    with pytest.raises(ManifestError, match="incomplete"):
        validate_class_manifest({
            "status": "class_manifest_incomplete",
            "expected_total_classes": 31,
            "classes": [],
            "crops": ["Coconut", "Arecanut", "Banana", "Paddy", "Pepper", "Cardamom"],
        })


def test_manifest_rejects_group_leakage(tmp_path):
    csv_path = tmp_path / "manifest.csv"
    pd.DataFrame([
        {"image_path": "a.jpg", "crop": "Paddy", "label": "X", "symptom_text": "", "source": "s", "group_id": "plant-1", "split": "train"},
        {"image_path": "b.jpg", "crop": "Paddy", "label": "X", "symptom_text": "", "source": "s", "group_id": "plant-1", "split": "test"},
    ]).to_csv(csv_path, index=False)
    with pytest.raises(ManifestError, match="leaks across splits"):
        validate_dataset_manifest(csv_path, require_files=False)
