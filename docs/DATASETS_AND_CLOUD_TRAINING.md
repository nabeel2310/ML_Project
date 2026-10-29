# Dataset acquisition, training, evaluation, and deployment plan

## Status and non-negotiable facts

This repository is a code foundation, not a trained diagnostic model. No dataset has been downloaded, no 31-class manifest has been approved, no GPU training run has been completed, and no model accuracy is claimed. The class manifest must be completed from a reviewed disease taxonomy before training. Never invent class labels to fill the target count.

The target scope is six crops: coconut, arecanut, banana, paddy/rice, black pepper, and cardamom. A 31-class target is a planning constraint only until each class has a verified name, crop, symptom/disease definition, and source evidence.

## Open dataset acquisition strategy

Use a multi-source dataset, but preserve source, license, and capture context for every record. Public does not automatically mean open-licensed or suitable for commercial redistribution. Review each dataset's current license and terms before downloading, using, or republishing it. Keep source images in object storage or a local data volume; Git should contain only manifests and scripts, not image corpora.

### Candidate sources to review

These are discovery starting points, not a promise that each source contains the target crops or can be redistributed. Inspect class coverage, license, image provenance, and label quality before inclusion.

| Source | Likely value | Limitations / required checks |
|---|---|---|
| PlantVillage (GitHub) — https://github.com/spMohanty/PlantVillage-Dataset | Large, widely used leaf-disease baseline; useful for pipeline smoke tests and some overlapping crop classes | Many images are clean-background/lab-style; coverage does not match all six target crops; verify license and exact labels |
| PlantDoc (GitHub) — https://github.com/pratikkayal/PlantDoc-Dataset | More in-the-wild plant disease images and varied backgrounds | No guarantee of target crop coverage; inspect annotation quality, license, duplicates |
| Paddy Doctor (Kaggle) — https://www.kaggle.com/competitions/paddy-disease-classification | Rice/paddy disease image classification benchmark | Competition terms apply; follow data terms and avoid leakage from related images |
| Plant Pathology 2020 (Kaggle) — https://www.kaggle.com/c/plant-pathology-2020-fgvc7 | Apple leaf disease benchmark for general representation-learning experiments | Apple-only; not a target-class source; terms apply |
| Plant Pathology 2021 (Kaggle) — https://www.kaggle.com/c/plant-pathology-2021-fgvc8 | Apple disease multi-label benchmark for general representation learning | Apple-only; not a target-class source; terms apply |
| Hugging Face Datasets Hub — https://huggingface.co/datasets | Search for crop-specific research datasets and dataset cards | Licenses and labels vary; verify the actual source and license |
| Zenodo — https://zenodo.org | Research datasets with DOI, paper, and often license metadata | Verify license, crop, geographic provenance, and reuse permission |
| Mendeley Data — https://data.mendeley.com | Crop-specific datasets attached to research publications | Verify license and label definitions; availability does not guarantee redistribution rights |
| Kaggle Datasets — https://www.kaggle.com/datasets | Potential crop-specific coconut, arecanut, banana, pepper, and cardamom collections | Terms vary; require documented license, stable source URL, and label audit; do not bypass access controls |

### Search and acquisition order

1. Search each target crop plus terms such as leaf disease dataset, field image, symptom, fungal, bacterial, pest, and relevant local disease names.
2. Prioritize official research repositories, paper-linked archives, government/university extension sources, and datasets with explicit reuse licenses.
3. Build a source register with source ID, canonical URL, paper/DOI, license, access date, crop(s), labels, image count, geography, capture setting, annotation method, caveats, and SHA-256 checksum of downloaded archive.
4. Do not use a source until its terms have been reviewed. Record permission unclear and exclude it until clarified.
5. Download only via the publisher/platform's permitted mechanism (e.g. official API/CLI). Store credentials in cloud secret storage, never in Git.
6. Keep raw files immutable. Create derived/cleaned versions separately and record transformation scripts.

### Required target-domain data

General plant datasets cannot substitute for target-domain examples. Actively collect licensed, consented field images for coconut, arecanut, banana, paddy, pepper, and cardamom across varieties, growth stages, lighting, weather, phones/cameras, healthy leaves, disease severity, and look-alike symptoms. Include healthy samples and non-disease stressors. Ask a qualified agronomist/plant pathologist to verify labels. For each sample capture crop, disease/class, coarse region only where permitted, date/season if permitted, plant part, severity, capture device, annotator, confidence, and source/consent.

Never label a class solely from an AI model prediction. Keep ambiguous cases as needs_review, not as training labels.

## Data quality, deduplication, and leakage prevention

- Complete config/crop_classes.json only after an expert-reviewed taxonomy is approved.
- Maintain one CSV manifest using data/manifest.example.csv as the schema template; image files stay outside Git.
- Hash files and use perceptual hashes/embedding similarity to detect exact and near duplicates across all sources.
- Group splits by plant, field, farm, collection session, original image lineage, and source where possible. Do not randomly split near-identical crops into train and test.
- Keep a locked test set from independent farms/regions or a later collection period. Do not use it for tuning.
- Create train/validation/test partitions before augmentation. Augmented variants must never cross partitions.
- Audit class counts, missing files, corrupt images, label conflicts, duplicate clusters, class/source imbalance, and class coverage before training.
- Store split IDs and manifest checksums so experiments can be reproduced.
- Use external datasets with different taxonomies only for compatible pretraining or explicitly mapped classes; never silently merge different label definitions.

## Cloud training plan

### Recommended practical path

1. Pipeline smoke test: a small representative subset on a free/low-cost notebook GPU. Verify data loading, loss decreasing, checkpoint save/resume, and evaluation end-to-end.
2. Baseline: train a single pretrained backbone first (ConvNeXt or Swin) at 224/256 resolution. This gives a cost/performance reference before attempting a large fusion model.
3. Scale-up: use a persistent GPU VM or managed GPU notebook with a suitable NVIDIA GPU and enough VRAM. The planned dual ConvNeXt-Large + Swin-Large at 384px plus DeBERTa-v3-Large may require substantial VRAM; use gradient accumulation, mixed precision, activation checkpointing, and freeze/warm-up stages. Benchmark memory on a small run before choosing hardware.
4. Multimodal phase: add symptom text only after image-only baselines are stable and a text field is consistently collected. Train with missing-text examples so the model is not brittle when descriptions are absent.
5. Experiment tracking: log Git commit, config, dataset-manifest hash, class manifest version, GPU type, package versions, seed, train/validation curves, per-class metrics, calibration, and checkpoint artifact URI. Weights & Biases, MLflow, or TensorBoard are options; check terms/privacy before uploading any data.
6. Checkpointing: save best validation checkpoint, last checkpoint, optimizer/scaler/scheduler state, epoch, RNG state, class map, and preprocessing config. Write to persistent/object storage, not ephemeral notebook disk. Test resume.
7. Evaluation: report macro-F1, balanced accuracy, per-class precision/recall/F1, confusion matrix, top-1 accuracy, calibration (ECE/Brier), abstention/unknown handling, latency, and subgroup results by source/region/device. Accuracy alone is insufficient.
8. Independent review: evaluate on a locked, independently collected field test set. Compare against an expert-labeled subset and baseline models. Do not claim >99% until measured on a leakage-resistant independent test set and reproduced.
9. Model card: document intended use, exclusions, training data provenance/licensing, performance by class and domain, failure modes, uncertainty behavior, and limits of treatment advice.
10. Deployment: export a versioned model artifact to private object storage or model registry; deploy a FastAPI inference service; store only minimal prediction logs with appropriate consent and retention. Use RLS and least-privilege keys for Supabase. Never expose service-role keys in a browser or mobile app.

### Compute choices to compare

- Google Colab / Colab Pro: fastest notebook prototype; sessions and local disks can be ephemeral, so checkpoint to Drive/object storage frequently.
- Kaggle Notebooks: useful for permitted Kaggle datasets and bounded experiments; internet, GPU, runtime, and dataset rules may constrain workflow.
- RunPod or another GPU cloud: flexible persistent GPU instances; compare current GPU availability, VRAM, storage/network costs, and shutdown billing before launching.
- Cloud provider GPU VM (AWS/GCP/Azure): more control over IAM, networking, storage, and repeatability, but more setup and cost management.
- Lightning AI or similar managed ML workspaces: convenient experiment environments; validate current plan limits and persistence features.

Do not commit to a specific GPU or budget until the image count, batch size, resolution, model configuration, and trial throughput are benchmarked. Use a cost cap, alerts, automatic shutdown, and a written experiment budget.

## Proposed experiment ladder

A. Data/label audit and leakage checks.
B. Lightweight baseline (e.g. EfficientNet/ConvNeXt-Tiny) on the actual target classes.
C. Single large backbone at 224/384 resolution.
D. Compare ConvNeXt-Large and Swin-Large independently.
E. Ensemble or fused dual backbone only if validation gains justify cost.
F. Optional symptom text model and missing-modality robustness.
G. Sub-center classifier, supervised contrastive loss, EMA, and calibration ablations.
H. External, locked field evaluation and deployment hardening.

Run controlled ablations. Do not add every technique simultaneously; otherwise it is impossible to tell what helps.

## Acceptance gates before production

- [ ] All 31 classes defined and reviewed; healthy/unknown/uncertain handling specified.
- [ ] Dataset source register includes license and provenance for every source.
- [ ] Manifest has no missing/corrupt images or unresolved label conflicts.
- [ ] Duplicate and group-leakage audit completed.
- [ ] Locked independent test set documented and protected from tuning.
- [ ] Reproducible training and resume test passed.
- [ ] Per-class metrics and calibration reviewed by domain expert.
- [ ] Uncertainty threshold and consult-an-agronomist fallback tested.
- [ ] Treatment recommendations grounded in reviewed extension/agronomy sources and location-appropriate regulations; no unsupported chemical dosage advice.
- [ ] Security, privacy, Supabase RLS, retention, and deployment health checks completed.
- [ ] Model card and versioned artifacts published.

## Current blockers

The class manifest is deliberately incomplete, and no approved dataset manifest, downloaded image corpus, GPU run, trained checkpoint, measured accuracy, or production deployment exists yet. These are explicit gates, not completed work.