# Additional Open Dataset Plan for the Six-Crop Disease Model

**Branch:** `feature/crop-disease-foundation`  
**Goal:** Build a licensed, deduplicated, source-traceable image corpus for Coconut, Arecanut, Banana, Paddy/Rice, Black Pepper (`Piper nigrum`), and Cardamom. The project target is 5,000–10,000 *unique, usable original images per crop* where public data permits. This is a target, not a claim that every crop already meets it.

## Dataset leads and current status

| Crop | Public dataset lead | Reported size | Licence/access notes | Status for this project |
|---|---|---:|---|---|
| Coconut | [Coconut Tree Disease Dataset, Mendeley Data](https://data.mendeley.com/datasets/gh56wbsnj5/1) | 5,798 labelled images across 5 diseases | CC BY 4.0 reported on the dataset page; cite the dataset/article | Meets the 5k crop-level target, but class balance is uneven (smallest class 470). |
| Banana | [Banana Leaves Imagery Dataset paper and data availability](https://www.nature.com/articles/s41597-025-04456-4) | 11,767 images: 3,339 healthy, 3,496 Black Sigatoka, 4,932 Fusarium Wilt Race 1 | Open-access data descriptor; confirm the linked data repository's exact reuse terms before distribution | Meets crop-level target. Disease coverage is limited to the labels supplied by this dataset. |
| Paddy/Rice | [Paddy Disease Classification / Paddy Doctor data card](https://huggingface.co/datasets/Project-AgML/paddy_disease_classification) and [Paddy Doctor dataset page](https://paddydoc.github.io/dataset/) | 10,407 in one curated classification release; paper site reports 16,225 cleaned annotated images for a broader dataset | Check the exact source terms and whether RGB and infrared images are mixed before training | Meets crop-level target. Choose one canonical source version and avoid mixing RGB/IR as if they were equivalent. |
| Black Pepper (`Piper nigrum`) | [Roboflow Piper nigrum Leaf Disease dataset](https://universe.roboflow.com/pipernigrum/piper-nigrum-leaf-disease) | Search result reports version 3 with 5,177 images; page also displays a smaller image count, so this must be verified after download | CC BY 4.0 is reported on the project page; confirm version-specific export terms and exact count | Candidate only; verify the actual export and deduplicate before counting it toward target. Do **not** substitute bell/chilli pepper (`Capsicum`) images for black pepper. |
| Arecanut | [Arecanut Disease Detection Dataset, GitHub](https://github.com/BeenaPradeep/ARECANUT-DATASET) | Approximately 300 field images, 6 folders/classes | Public repository; inspect its licence and class definitions before reuse | Does **not** meet the target. Requires more public sources and/or new field collection with expert labels. |
| Cardamom | [Cardamom-Dataset-2021, GitHub](https://github.com/sunilchinnahalli/Cardamom-Dataset-2021) | Repository mentions 1,724 images in a shared Drive file; availability is also described as “upon request” | Access and reuse licence are not clear from the repository summary; obtain permission/terms before use | Does **not** meet the target. Requires permission, additional sources, and/or new field collection. |

## Important interpretation of the 5k–10k target

- The target above is **per crop in total**, not 5,000–10,000 images for every disease class. The latter would require a much larger corpus.
- A reported total does not guarantee 5,000 usable originals after duplicate removal, broken-image checks, label review, and split isolation.
- Augmented copies do not count as unique original images. Augmentation is applied to training data only after splits are made.
- A dataset with a similar crop name is not necessarily the same crop: **black pepper is Piper nigrum; bell/chilli pepper is Capsicum**. Never merge their labels.
- Preserve source, original URL/ID, label, licence, capture context, and duplicate-group ID in the manifest.
- Split by source, plant/leaf identity, or capture session where possible. Do not let near-identical photographs of one leaf appear across train and test.
- Do not merge labels merely because names sound similar. A plant pathologist/agronomist should approve the final disease taxonomy and any cross-dataset label mapping.

## Acquisition workflow

1. Download each source into a separate directory under `data/external/<source_slug>/`; do not commit image archives or credentials to Git.
2. Record the source URL, download date, version, licence, citation, and raw image count in `data/source_registry_template.csv`.
3. Validate image readability, dimensions, labels, and crop identity.
4. Compute cryptographic hashes and perceptual hashes to find exact and near-duplicate images across datasets.
5. Review ambiguous labels; keep unknown/uncertain images out of supervised disease classes.
6. Build one canonical manifest, retaining the original source label and the reviewed mapped label.
7. Create leakage-resistant train/validation/test splits **before augmentation**. Keep a source-held-out test set where possible.
8. Report per-crop and per-class counts after cleaning. If Arecanut or Cardamom remains below target, report the shortfall honestly and prioritize field data collection rather than padding counts with synthetic copies.

## Current decision

The public sources found so far provide promising crop-level coverage for Coconut, Banana, and Paddy/Rice. Black Pepper has a candidate source whose downloadable version/count must be checked. Arecanut and Cardamom currently have a substantial data gap. Therefore, it is **not yet accurate to claim that all six crops have 5,000–10,000 clean, licensed images ready for final training**. Final training should begin only after the manifest, licences, class taxonomy, and leakage checks are reviewed.
