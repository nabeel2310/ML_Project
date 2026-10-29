import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_crop_manifest_has_expected_crop_names():
    manifest = json.loads((ROOT / "config" / "crop_classes.json").read_text(encoding="utf-8"))
    assert manifest["expected_total_classes"] == 31
    assert manifest["crops"] == ["Coconut", "Arecanut", "Banana", "Paddy", "Pepper", "Cardamom"]


def test_class_manifest_is_not_misrepresented_as_complete():
    manifest = json.loads((ROOT / "config" / "crop_classes.json").read_text(encoding="utf-8"))
    # Until the exact approved 31 labels are entered, training should remain blocked.
    assert manifest["status"] == "class_manifest_incomplete"
    assert manifest["classes"] == []
