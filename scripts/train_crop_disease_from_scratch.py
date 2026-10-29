#!/usr/bin/env python3
"""Train a crop-disease classifier from random initial weights (no pretrained model).

Expected dataset layout:
data/processed/crop_disease/
  Arecanut/
    Healthy_Leaf/*.jpg
    Mahali_Koleroga/*.jpg
  Cardamom/
    Healthy/*.jpg
    Leaf_Blight/*.jpg
  Coconut/
    Healthy/*.jpg
    ...
The final label is "<Crop>__<Disease>". Do not place the same image in multiple splits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def scan_dataset(root: Path, min_images: int = 3):
    """Scan root/<crop>/<disease>/<image>, skip unreadable images and exact duplicates."""
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {root}\n"
            "Create data/processed/crop_disease/<Crop>/<Disease>/ and put images inside."
        )
    records, seen_hashes, skipped = [], set(), 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel = path.relative_to(root)
        if len(rel.parts) < 3:
            continue
        crop, disease = rel.parts[0], rel.parts[1]
        try:
            with Image.open(path) as im:
                im.verify()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen_hashes:
                skipped += 1
                continue
            seen_hashes.add(digest)
            records.append((path, f"{crop}__{disease}"))
        except (OSError, UnidentifiedImageError):
            skipped += 1

    counts = Counter(label for _, label in records)
    too_small = {k: v for k, v in counts.items() if v < min_images}
    if too_small:
        raise ValueError(
            "Each class needs at least "
            f"{min_images} unique readable images for train/validation/test splits. "
            f"Small classes: {too_small}"
        )
    if len(counts) < 2:
        raise ValueError(f"Need at least 2 classes; found {len(counts)}.")
    print(f"Found {len(records)} unique readable images across {len(counts)} classes.")
    print(f"Skipped {skipped} unreadable files or exact duplicate images.")
    return records, sorted(counts)


def stratified_split(records, seed: int, train_ratio: float, val_ratio: float):
    rng = random.Random(seed)
    by_class = {}
    for item in records:
        by_class.setdefault(item[1], []).append(item)
    train, val, test = [], [], []
    for label, items in sorted(by_class.items()):
        rng.shuffle(items)
        n = len(items)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        if n_train + n_val >= n:
            n_train, n_val = n - 2, 1
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


class CropDiseaseDataset(Dataset):
    def __init__(self, records, class_to_idx, transform):
        self.records = records
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label = self.records[index]
        with Image.open(path) as im:
            image = im.convert("RGB")
        return self.transform(image), self.class_to_idx[label]


class ScratchCNN(nn.Module):
    """Small CNN initialized randomly; intentionally uses no pretrained weights."""
    def __init__(self, num_classes: int):
        super().__init__()
        def block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(
            block(3, 32),
            block(32, 64),
            block(64, 128),
            block(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            for actual, predicted in zip(labels.cpu(), predictions.cpu()):
                matrix[actual, predicted] += 1
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "confusion_matrix": matrix.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/crop_disease"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/crop_disease_scratch"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if not (0.5 <= args.train_ratio < 1.0 and 0.0 < args.val_ratio < 0.5
            and args.train_ratio + args.val_ratio < 1.0):
        parser.error("Require train ratio >= 0.5, validation ratio > 0, and train + validation < 1.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    records, labels = scan_dataset(args.data_dir)
    train_records, val_records, test_records = stratified_split(
        records, args.seed, args.train_ratio, args.val_ratio
    )
    class_to_idx = {name: i for i, name in enumerate(labels)}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train_ds = CropDiseaseDataset(train_records, class_to_idx, train_transform)
    val_ds = CropDiseaseDataset(val_records, class_to_idx, eval_transform)
    test_ds = CropDiseaseDataset(test_records, class_to_idx, eval_transform)

    train_counts = Counter(label for _, label in train_records)
    sample_weights = [1.0 / train_counts[label] for _, label in train_records]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    loader_args = dict(batch_size=args.batch_size, num_workers=args.workers,
                       pin_memory=(device.type == "cuda"))
    train_loader = DataLoader(train_ds, sampler=sampler, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_args)

    model = ScratchCNN(num_classes=len(labels)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    history, best_val_loss = [], float("inf")
    best_path = args.output_dir / "best_model.pt"
    start = time.time()

    config = {
        "architecture": "ScratchCNN",
        "pretrained_weights": None,
        "initialization": "random",
        "seed": args.seed,
        "device": str(device),
        "image_size": args.image_size,
        "epochs_requested": args.epochs,
        "class_to_idx": class_to_idx,
        "split_counts": {"train": len(train_ds), "validation": len(val_ds), "test": len(test_ds)},
        "class_counts_all": dict(Counter(label for _, label in records)),
        "data_dir": str(args.data_dir),
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * targets.size(0)
            correct += (logits.argmax(1) == targets).sum().item()
            total += targets.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_result = evaluate(model, val_loader, criterion, device, len(labels))
        scheduler.step(val_result["loss"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "validation_loss": val_result["loss"],
            "validation_accuracy": val_result["accuracy"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | train loss {train_loss:.4f} "
            f"acc {train_acc:.3f} | val loss {val_result['loss']:.4f} "
            f"acc {val_result['accuracy']:.3f}"
        )
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if val_result["loss"] < best_val_loss:
            best_val_loss = val_result["loss"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
                "config": config,
                "epoch": epoch,
                "validation": val_result,
            }, best_path)

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_result = evaluate(model, test_loader, criterion, device, len(labels))
    per_class = {}
    matrix = test_result["confusion_matrix"]
    for label, idx in class_to_idx.items():
        tp = matrix[idx][idx]
        fp = sum(matrix[r][idx] for r in range(len(labels)) if r != idx)
        fn = sum(matrix[idx][c] for c in range(len(labels)) if c != idx)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1,
                            "support": sum(matrix[idx])}

    metrics = {
        "test_loss": test_result["loss"],
        "test_accuracy": test_result["accuracy"],
        "per_class": per_class,
        "confusion_matrix": matrix,
        "class_to_idx": class_to_idx,
        "best_checkpoint_epoch": checkpoint["epoch"],
        "training_seconds": round(time.time() - start, 2),
        "note": "ScratchCNN trained from random initialization; test metrics are only meaningful if dataset splits are leakage-free and labels are correct.",
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nBest checkpoint: {best_path}")
    print(f"Test accuracy: {test_result['accuracy']:.4f}")
    print(f"Metrics: {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
