# Train the crop-disease model from scratch

This pipeline initializes a small CNN with random weights. It does **not** load ImageNet or any other pretrained weights.

## 1. Prepare the data

Download only datasets you can access and are permitted to reuse. The research leads in `docs/ADDITIONAL_DATASET_PLAN.md` are not all confirmed public downloads; some require contacting authors. Do not assume the reported image counts are available or licensed for redistribution.

Arrange extracted images like this:

```text
data/processed/crop_disease/
├── Arecanut/
│   ├── Healthy_Leaf/
│   ├── Mahali_Koleroga/
│   └── Yellow_Leaf_Disease/
├── Banana/
│   ├── Healthy/
│   └── Black_Sigatoka/
├── Cardamom/
│   ├── Healthy/
│   └── Leaf_Blight/
├── Coconut/
├── Paddy/
└── Black_Pepper/
```

Every image must be inside a crop folder and then a disease/class folder. The model label is made from both folder names, for example `Arecanut__Healthy_Leaf` and `Cardamom__Healthy`, so identical disease names in different crops are kept separate. Use consistent, verified labels; do not invent missing classes or treat product/spice photos as disease-leaf images.

Keep the original downloads separately. Maintain the source registry with the dataset URL, licence, download date, original label, mapped label, and any permission notes. Do not add datasets that prohibit your intended use.

## 2. Install dependencies

Use Python 3.10 or newer in a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision pillow
```

## 3. Run training

From the repository root:

```bash
python scripts/train_crop_disease_from_scratch.py --data-dir data/processed/crop_disease --epochs 30 --batch-size 32
```

If training is slow or memory is limited, try `--batch-size 16` or `--image-size 160`. On a CUDA-capable machine, PyTorch will use the GPU automatically; otherwise it uses the CPU.

## 4. Outputs

Training creates `outputs/crop_disease_scratch/` containing:

- `best_model.pt`: checkpoint selected by validation loss
- `config.json`: classes, split counts, and model setup
- `history.json`: epoch-by-epoch training and validation metrics
- `metrics.json`: held-out test accuracy, per-class precision/recall/F1, and confusion matrix

The script removes byte-identical files before splitting and uses a stratified random split. It cannot detect near-duplicates or photos from the same plant/session. Before relying on test scores, check for duplicate/related images across sources and ensure images from the same plant, video, or capture session do not cross splits. Keep the test set untouched while tuning the model.

## Important limitations

- Training from scratch typically needs more varied labelled images than fine-tuning a pretrained model. A small CNN may underperform, especially on rare classes.
- A reported dataset total is not the same as the number of accessible, unique, correctly labelled images.
- Only classes actually present in the prepared folders can be learned. Do not describe the model as a 31-class or six-crop model unless the verified dataset covers those classes.
- The script is a baseline, not a plant-health diagnosis tool. Evaluate on field images from locations and conditions not represented in training before any real-world use.
