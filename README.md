# Multimodal Crop Disease Detection

Research and engineering project for crop disease screening from an image and an optional natural-language symptom description, targeting **Coconut, Arecanut, Banana, Paddy/Rice, Black Pepper, and Cardamom**.

> **Reality/status:** this repository is a foundation and project plan, not a completed diagnostic product. No full dataset corpus, approved 31-class taxonomy, trained checkpoint, measured accuracy, completed GPU training run, or production deployment is currently included. A target of >99% is an aspiration, never a guarantee. Only report measured results from a locked, independent, leakage-checked test set.

## Project goals

- Build a reproducible image classification baseline, then evaluate whether multimodal fusion improves results.
- Eventually evaluate a ConvNeXt-Large + Swin-Large image encoder, optional DeBERTa-v3-Large symptom encoder, cross-attention/gated fusion, sub-center cosine classifier, EMA, and supervised contrastive learning.
- Provide a robust vision-only path when symptom text is missing.
- Return calibrated confidence/uncertainty and abstain when the evidence is weak.
- Keep source provenance, licenses, experiment configuration, and model versions auditable.
- Deploy only after expert evaluation, privacy/security review, and independent field testing.

## Current status and implementation map

| Area | Status |
|---|---|
| Git branch and repository foundation | In progress on `feature/crop-disease-foundation`; `main` is kept separate |
| Configuration, environment template, initial schema | Added; real database migration has not been applied/verified |
| Manifest validation and leakage-oriented checks | Foundation added; requires real data and end-to-end validation |
| Vision/text model components and loss/EMA helpers | Foundation added; not yet validated in a full training run |
| Dataset source research and provenance templates | Added in docs and CSV template; datasets have not been assembled |
| Exact target classes | **Blocked:** `config/crop_classes.json` intentionally has no final classes |
| Training pipeline and reproducible GPU run | Not completed |
| Trained checkpoint / accuracy claim | None |
| Inference API, UI, monitoring, production deployment | Not completed |

## Read this first

1. [Open dataset catalog](docs/OPEN_DATASET_CATALOG.md) — crop-specific dataset leads and licensing checks.
2. [Dataset acquisition and cloud training roadmap](docs/DATASETS_AND_CLOUD_TRAINING.md) — complete workflow and acceptance gates.
3. [Training configuration template](config/training.example.yaml) — baseline experiment settings.
4. [Dataset source registry template](data/source_registry_template.csv) — source provenance and license tracking.
5. [Example data manifest](data/manifest.example.csv) — metadata schema for image records.
6. [Supabase migration](supabase/migrations/0001_initial_schema.sql) — initial database tables and RLS baseline.

## End-to-end work plan

### Phase 0 — Scope, taxonomy, and risk decisions

- [ ] Confirm the exact 31 target labels with a qualified plant pathologist/agronomist; 31 is a target count, not an approved taxonomy.
- [ ] For every label, define crop, accepted synonyms, plant part, disease/healthy/stress status, visible evidence, common look-alikes, and source references.
- [ ] Decide how to represent healthy plants, nutrient/water stress, pest damage, unknown crops, out-of-scope diseases, poor-quality images, and uncertain diagnoses.
- [ ] Define intended use as decision support/screening, not a replacement for an expert diagnosis.
- [ ] Decide initial target geography and language(s); disease prevalence and treatment recommendations are location-dependent.
- [ ] Have a domain expert approve the taxonomy before building the final class map.

### Phase 1 — Find, license, acquire, and register a large dataset

Use the [dataset catalog](docs/OPEN_DATASET_CATALOG.md) as a starting point. Candidate sources include:
- [Coconut Tree Disease Dataset (Mendeley Data)](https://data.mendeley.com/datasets/gh56wbsnj5/1)
- [Arecanut dataset repository](https://github.com/BeenaPradeep/ARECANUT-DATASET)
- [Paddy Disease Classification (Kaggle)](https://www.kaggle.com/competitions/paddy-disease-classification)
- [Rice Leaf Bacterial and Fungal Disease Dataset (Mendeley Data)](https://data.mendeley.com/datasets/hx6f852hw4/2)
- [Banana Leaves Imagery Dataset](https://www.nature.com/articles/s41597-025-04456-4)
- [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) and [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset) for compatible classes/representation-learning experiments
- [Hugging Face Datasets](https://huggingface.co/datasets), [Zenodo](https://zenodo.org), [Mendeley Data](https://data.mendeley.com), and [Kaggle Datasets](https://www.kaggle.com/datasets) for further source discovery

**These are leads, not an assertion that every dataset is open-licensed, target-compatible, or downloaded.** Review each source's current license, terms, paper, class definitions, and redistribution rules. Never bypass access controls. If permission is unclear, exclude the source until resolved. Do not infer a license from the fact that a dataset is publicly visible.

- [ ] Search each crop and disease name, including regional/local disease terms.
- [ ] Prioritize source repositories with a paper/DOI, explicit license, and field provenance.
- [ ] Search especially for Black Pepper and Cardamom; current known coverage is a gap and may require new expert-labeled field collection.
- [ ] Register each source in `data/source_registry_template.csv`: URL, DOI/paper, license, access date, crop/classes, counts, geography, capture setting, annotation method, checksum, and inclusion decision.
- [ ] Use only the official permitted download mechanism; store download credentials in secrets, never in Git.
- [ ] Keep raw data immutable and separate from cleaned/derived datasets.
- [ ] Do not put image corpora or private datasets in Git. Use persistent disk or object storage.
- [ ] Record attribution and comply with licenses for research, commercial use, redistribution, and derivative datasets.

### Phase 2 — Collect target-domain field data

General datasets often have different backgrounds, crops, regions, or cameras from the real deployment environment. Obtain consent and permission for new field images where needed.

- [ ] Sample all six crops, healthy plants, confirmed diseases, nutrient/water stress, pest damage, and common look-alikes.
- [ ] Cover different farms, districts/regions, seasons, varieties, plant ages, severity, plant parts, cameras, lighting, weather, and distances.
- [ ] Ask trained annotators and a qualified plant pathologist/agronomist to review labels.
- [ ] Record only metadata that is needed and permitted: crop, class, plant part, severity, coarse region, capture context, annotator, confidence, consent/source, and group identifier.
- [ ] Keep uncertain samples in a review queue; never turn model guesses into ground truth automatically.
- [ ] Monitor class imbalance. Do not fix it by putting duplicated/augmented copies in the test set.

### Phase 3 — Clean, audit, and split the data

Use `data/manifest.example.csv` as a schema starting point and the validation utilities in `src/crop_disease/`.

- [ ] Create a complete manifest with valid paths, crop, exact label, source ID, optional symptom text, and group ID.
- [ ] Check missing files, corrupt images, invalid labels, unreadable files, conflicting labels, and class/source counts.
- [ ] Calculate file hashes and perceptual hashes; inspect likely near-duplicates across all sources.
- [ ] Track original image lineage. All augmented variants must stay in the same partition as the original.
- [ ] Split by plant/farm/plot/collection session/source lineage where possible—not by naive random image split.
- [ ] Lock a test set from independent farms/regions or a later collection period; never use it for tuning.
- [ ] Save split assignments, dataset manifest checksum, class map version, and audit report.
- [ ] Keep a separate out-of-domain/unknown set to test safe rejection.
- [ ] Run manifest and leakage checks in CI or as a pre-training gate.

### Phase 4 — Cloud infrastructure and budget control

Candidate platforms:
- [Google Colab](https://colab.research.google.com/) for a quick prototype, with checkpoints copied to persistent storage.
- [Kaggle Notebooks](https://www.kaggle.com/code) for experiments compatible with Kaggle's data, internet, and runtime terms.
- [RunPod](https://www.runpod.io/) or a comparable GPU cloud for configurable GPU instances and persistent volumes.
- [AWS](https://aws.amazon.com/), [Google Cloud](https://cloud.google.com/), or [Microsoft Azure](https://azure.microsoft.com/) for managed infrastructure, IAM, storage, and production networking.
- [Lightning AI](https://lightning.ai/) or equivalent managed ML workspaces, after checking current plan limits and data terms.

- [ ] Choose based on measured VRAM use, throughput, storage bandwidth, data location, runtime limits, and total cost—not marketing specifications alone.
- [ ] Start with a small GPU smoke test and a single backbone.
- [ ] Set a hard budget, cost alerts, persistent checkpoint destination, and automatic shutdown before paid training.
- [ ] Store secrets in the platform's secret manager/environment configuration.
- [ ] Keep datasets and checkpoints in persistent/object storage; notebook-local disks can disappear.
- [ ] Record GPU type, software versions, commit SHA, seed, dataset checksum, config, and run ID.
- [ ] Benchmark batch size, image size, training throughput, and memory before selecting expensive hardware.

The dual ConvNeXt-Large + Swin-Large at 384px plus DeBERTa-v3-Large can be resource-intensive. Use gradient accumulation, mixed precision, activation checkpointing, and staged freezing if needed. Do not assume that the largest architecture will be the most accurate or cost-effective.

### Phase 5 — Train in controlled experiments

1. **Smoke test:** verify reading a tiny approved subset, forward/backward passes, metrics, checkpoint save, and resume.
2. **Baseline:** train a lightweight pretrained model such as ConvNeXt-Tiny at 224px.
3. **Single-backbone comparisons:** compare ConvNeXt and Swin separately using the same split and evaluation procedure.
4. **Scale-up:** test larger backbones/resolutions only if baseline evidence justifies it.
5. **Fusion:** introduce cross-attention/gated multimodal fusion only after the image baseline is stable.
6. **Text:** use DeBERTa-v3-Large only when symptom text is reliably collected; train with missing text and test the vision-only path.
7. **Ablations:** compare sub-center classification, label smoothing, supervised contrastive loss, EMA, augmentation policies, and calibration independently.
8. **Reproducibility:** fix and record random seeds; store config, environment/package versions, class mapping, split hash, and Git commit.
9. **Checkpointing:** save best validation checkpoint and resumable last checkpoint with model, optimizer, scheduler, scaler, epoch, RNG states, class map, and preprocessing settings.
10. **Experiment tracking:** use TensorBoard, MLflow, or Weights & Biases; do not upload private data or sensitive metadata without approval.

Use `config/training.example.yaml` as a starting point, not a guaranteed optimal configuration. Validate it against the final dataset and hardware.

### Phase 6 — Evaluation and scientific validation

- [ ] Evaluate only on the locked test set after model and thresholds are frozen.
- [ ] Report top-1 accuracy, macro-F1, balanced accuracy, per-class precision/recall/F1, confusion matrix, calibration (ECE/Brier), inference latency, and memory.
- [ ] Break results down by crop, disease, healthy/stress class, source, region, camera, severity, and image quality where sample size allows.
- [ ] Measure false negatives for serious diseases and false positives that could prompt unnecessary treatment.
- [ ] Test images with missing text, misleading text, low light, cluttered backgrounds, multiple leaves, and unseen diseases.
- [ ] Set an abstention threshold and return “uncertain / seek expert review” when confidence is low or the input is out of scope.
- [ ] Compare against simple baselines and, where possible, expert assessments.
- [ ] Re-run final evaluation from a reproducible commit and preserve the report/artifact.
- [ ] Never claim >99% unless it is measured and independently reproduced on representative, leakage-resistant data.

### Phase 7 — Safe diagnosis and treatment guidance

- [ ] Tie advice to reviewed agricultural extension materials or qualified expert guidance for the user's region.
- [ ] Show the predicted class, confidence/uncertainty, visual evidence where validated, and next steps.
- [ ] Distinguish image classification from confirmed diagnosis.
- [ ] Avoid unsupported pesticide/chemical dosage advice; respect local label and regulatory requirements.
- [ ] Provide a human-review path for uncertain or high-impact cases.
- [ ] Maintain source references and review dates for recommendation content.

### Phase 8 — Supabase and backend

The project URL currently configured is `https://tnvmjahcdovffsgrajkm.supabase.co`.

- [ ] Apply `supabase/migrations/0001_initial_schema.sql` through the Supabase SQL Editor or a secured migration workflow.
- [ ] Verify the schema and RLS policies in the Supabase dashboard; migration application has not been confirmed by this repository setup.
- [ ] Add only the narrow policies needed for authenticated use; do not enable broad public read/write access.
- [ ] Use a publishable/anon key only with appropriate RLS; keep service-role and database passwords server-side.
- [ ] Configure real secrets in local `.env` or deployment secret storage only; never commit secrets or paste them into chat.
- [ ] Test unauthorized access, cross-user data isolation, deletion/retention, and failure paths.
- [ ] Store minimal prediction logs and avoid retaining uploaded images unless necessary and consented.
- [ ] Decide whether images stay ephemeral or are stored privately with signed access and retention limits.

### Phase 9 — API, user interface, deployment, monitoring

- [ ] Build an inference API with input validation, image size/type limits, timeout and rate limits.
- [ ] Add health/readiness checks, structured logs, versioned model loading, and clear error messages.
- [ ] Build a simple upload UI with crop selection, optional symptom description, result, uncertainty, and expert referral.
- [ ] Test CPU fallback or explicitly document GPU-only hosting requirements.
- [ ] Package dependencies reproducibly and scan for known vulnerabilities.
- [ ] Deploy a staging environment before production.
- [ ] Monitor latency, failures, drift, class/source distribution, and expert feedback without logging unnecessary personal data.
- [ ] Version model artifacts and keep rollback capability.
- [ ] Retrain only with reviewed labels and a documented evaluation gate.

### Phase 10 — Final release acceptance checklist

- [ ] Expert-approved 31-class taxonomy with healthy, stress, unknown, and uncertain behavior.
- [ ] Every included dataset has recorded provenance and a reviewed license.
- [ ] Target crops are adequately represented; gaps are explicitly documented.
- [ ] Duplicate, corruption, label-quality, and split-leakage audits passed.
- [ ] Reproducible training run and checkpoint resume test passed.
- [ ] Locked independent test evaluation and per-class metrics reviewed by a domain expert.
- [ ] Calibration, abstention, vision-only fallback, and out-of-domain behavior tested.
- [ ] Model card, data statement, experiment report, and versioned artifact published.
- [ ] Supabase RLS, secrets, privacy, API security, monitoring, and rollback tested.
- [ ] No unsupported accuracy or treatment claims.

## Setup

Python 3.11 is recommended. Install a PyTorch build compatible with your operating system and GPU using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/), then install the project requirements.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in only the values needed locally. Never commit `.env`, image corpora, model weights, Supabase service-role keys, or database passwords.

## Data manifest

Required/expected fields are described in `data/manifest.example.csv`. Each record should link to a file and its exact approved class, crop, source, split/group metadata, and optional symptom text. Keep data files outside Git and store their provenance in the manifest/registry.

Before any training, complete `config/crop_classes.json` with the expert-approved taxonomy and validate the manifest. An empty class list is an intentional blocker.

## Supabase

Project URL: `https://tnvmjahcdovffsgrajkm.supabase.co`. The database connection template in `.env.example` has a password placeholder. Replace it only in a local secret file or secure secret manager. The SQL migration has been authored but is **not confirmed applied**. Verify the actual schema and policies in the Supabase dashboard before connecting an application.

## Repository layout

- `config/crop_classes.json` — target crop/class manifest placeholder
- `config/training.example.yaml` — training experiment template
- `data/manifest.example.csv` — example data manifest
- `data/source_registry_template.csv` — dataset provenance/license register
- `docs/OPEN_DATASET_CATALOG.md` — dataset leads and caveats
- `docs/DATASETS_AND_CLOUD_TRAINING.md` — acquisition/training/deployment roadmap
- `src/crop_disease/` — package, data validation, model components, losses, EMA
- `supabase/migrations/` — initial schema and RLS baseline
- `tests/` — lightweight automated tests
- `.github/workflows/ci.yml` — basic CI checks

## Contribution and experiment records

Every dataset or model change should record source/licensing implications, manifest checksum, class map version, split definition, Git commit, configuration, seed, environment, hardware, metrics, and artifact URI. Keep large datasets/checkpoints in managed storage, not Git.