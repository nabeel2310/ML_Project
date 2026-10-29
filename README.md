# Multimodal Crop Disease Detection

Research prototype for image-based crop disease classification across Coconut, Arecanut, Banana, Paddy, Pepper, and Cardamom, with optional symptom text.

## Status

The repository foundation is being built in milestones. **No trained model weights or validated accuracy claims are included.** A target such as 99% accuracy is an aspiration, not a guarantee; performance must be measured on a held-out, leakage-checked test set.

## Milestones

1. Repository foundation, configuration, and secure database schema.
2. Dataset manifest, validation, duplicate/leakage checks, and split generation.
3. Vision/text model and training pipeline (AMP, gradient accumulation, EMA, validation).
4. Evaluation, calibration, and vision-only fallback tests.
5. Inference API/demo and optional Supabase logging.
6. Reproducible deployment and documentation.

## Setup

Python 3.11 is recommended. Install a PyTorch build compatible with your OS/GPU from the official PyTorch selector, then install the remaining dependencies.

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in only the keys needed for your environment. Never commit a real `.env`, model weights, dataset files, or Supabase secret/service-role key.

## Dataset manifest

Use a CSV manifest with at least:
- `image_path`: local path to an image
- `crop`: one of the six supported crops
- `label`: exact disease/healthy class name from the approved class list
- `symptom_text`: optional natural-language symptoms
- `source`: dataset/source identifier
- `group_id`: plant/plot/session identifier where available

The original specification calls for 31 target classes, but the exact class-to-crop mapping must be confirmed and reviewed before training. Do not silently assume similarly named labels from different datasets are equivalent. Split by plant/plot/source group where possible to reduce leakage.

## Supabase

Project URL configured in `.env.example`: `https://tnvmjahcdovffsgrajkm.supabase.co`. Apply the SQL migration in `supabase/migrations/` to this project using the Supabase SQL Editor or your migration workflow. The provided PostgreSQL connection string contains a password placeholder; replace it only in your local `.env` or a secure secret store, never in GitHub. Row Level Security (RLS) is enabled. Public users cannot read training samples or prediction logs by default. Add narrow policies only after deciding the application's authentication and data-access requirements.

Use publishable/anon keys only where RLS permits. Keep secret/service-role keys server-side in environment variables or deployment secrets. Never paste them into chat or commit them.

## Layout

- `config/crop_classes.json` — supported crops and explicit class-list placeholder
- `supabase/migrations/` — database schema and RLS baseline
- `src/crop_disease/` — application package
- `data/manifest.example.csv` — dataset manifest template
- `tests/` — basic configuration checks
