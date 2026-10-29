"""Train an image-only baseline on an extracted PlantVillage-style folder dataset.

Expected layout:
  data/external/newplantvillage/
    Apple___Apple_scab/*.jpg
    Apple___healthy/*.jpg
    ...

This is a general PlantVillage benchmark baseline, NOT the final six-crop model.
The source dataset does not cover the project's full target crop taxonomy.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/external/newplantvillage"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/plantvillage_baseline"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-images", type=int, default=0, help="Optional small smoke run; 0 uses all images.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("--epochs and --batch-size must be positive.")
    if not args.data_dir.is_dir():
        raise SystemExit(
            f"Dataset directory not found: {args.data_dir}\n"
            "Download and extract the dataset so each class has its own subfolder. "
            "See README.md section 'Run the PlantVillage baseline'."
        )

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = args.image_size
    train_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # Discover labels and image paths once; use separate transform instances for each split.
    discovery = datasets.ImageFolder(args.data_dir)
    if len(discovery.classes) < 2:
        raise SystemExit(f"Expected at least two class folders, found {len(discovery.classes)}.")
    targets = np.asarray(discovery.targets)
    indices = np.arange(len(discovery))
    if args.max_images:
        if args.max_images < 20:
            raise SystemExit("--max-images must be at least 20 for a smoke run.")
        rng = np.random.default_rng(args.seed)
        chosen = []
        for class_id in np.unique(targets):
            members = indices[targets == class_id]
            chosen.extend(rng.choice(members, size=min(len(members), max(1, args.max_images // len(discovery.classes))), replace=False).tolist())
        indices = np.asarray(sorted(set(chosen)), dtype=int)
        targets = np.asarray(discovery.targets)[indices]

    train_idx, holdout_idx = train_test_split(
        indices, test_size=0.20, random_state=args.seed, stratify=targets
    )
    val_idx, test_idx = train_test_split(
        holdout_idx, test_size=0.50, random_state=args.seed, stratify=np.asarray(discovery.targets)[holdout_idx]
    )

    train_ds = datasets.ImageFolder(args.data_dir, transform=train_tf)
    eval_ds = datasets.ImageFolder(args.data_dir, transform=eval_tf)
    train_loader = DataLoader(Subset(train_ds, train_idx.tolist()), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(Subset(eval_ds, val_idx.tolist()), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(Subset(eval_ds, test_idx.tolist()), batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=device.type == "cuda")

    # ImageNet transfer-learning baseline. We do not claim it matches the source repo's model.
    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except Exception as exc:
        raise SystemExit(
            "Could not load pretrained ResNet18 weights. Ensure internet access is enabled "
            "or cache the weights before running. Original error: " + str(exc)
        )
    model.fc = nn.Linear(model.fc.in_features, len(discovery.classes))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_val = -1.0
    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0
        for images, labels in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs} train"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += loss.item() * labels.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_count += labels.size(0)

        model.eval()
        val_correct = 0
        val_count = 0
        val_loss_sum = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(images)
                    loss = criterion(logits, labels)
                val_loss_sum += loss.item() * labels.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_count += labels.size(0)
        val_acc = val_correct / max(1, val_count)
        row = {
            "epoch": epoch,
            "train_loss": train_loss_sum / max(1, train_count),
            "train_accuracy": train_correct / max(1, train_count),
            "val_loss": val_loss_sum / max(1, val_count),
            "val_accuracy": val_acc,
        }
        history.append(row)
        print(json.dumps(row))
        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_to_idx": discovery.class_to_idx,
                "classes": discovery.classes,
                "image_size": size,
                "architecture": "torchvision.resnet18",
                "seed": args.seed,
            }, args.output_dir / "best_model.pt")

    # One final evaluation on the held-out random test split. This is not a field test;
    # near-duplicate images can still cross splits unless the dataset is deduplicated first.
    checkpoint = torch.load(args.output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="held-out test"):
            images = images.to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())

    report = classification_report(y_true, y_pred, labels=list(range(len(discovery.classes))),
                                   target_names=discovery.classes, output_dict=True, zero_division=0)
    summary = {
        "dataset_root": str(args.data_dir),
        "num_images_used": int(len(indices)),
        "num_classes": len(discovery.classes),
        "split_counts": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(discovery.classes)))).tolist(),
        "classes": discovery.classes,
        "seed": args.seed,
        "architecture": "torchvision.resnet18",
        "device": str(device),
        "epochs": args.epochs,
        "elapsed_seconds": round(time.time() - start, 2),
        "warning": "Random image split only. Deduplicate/group by plant, source, and capture session and run independent field evaluation before making real-world accuracy claims.",
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Finished. Test accuracy: {summary['test_accuracy']:.4%}")
    print(f"Checkpoint: {args.output_dir / 'best_model.pt'}")
    print(f"Metrics: {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
