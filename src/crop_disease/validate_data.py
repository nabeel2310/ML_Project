"""Validate class configuration and a CSV dataset manifest."""
import argparse
from pathlib import Path

from .data import (
    ManifestError,
    load_class_manifest,
    validate_class_manifest,
    validate_dataset_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True, help="Path to dataset CSV manifest")
    parser.add_argument("--skip-file-check", action="store_true", help="Validate metadata only")
    args = parser.parse_args()
    try:
        validate_class_manifest(load_class_manifest())
        frame = validate_dataset_manifest(args.manifest, require_files=not args.skip_file_check)
    except (ManifestError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Manifest valid: {len(frame)} rows; splits={frame['split'].value_counts(dropna=False).to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
