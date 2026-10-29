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
| Arecanut | [Kaggle dataset reported in a 2026 study](https://pmc.ncbi.nlm.nih.gov/articles/PMC13195169/); [Arecanut Disease Detection Dataset, GitHub](https://github.com/BeenaPradeep/ARECANUT-DATASET); [Arecanut disease-detection paper reporting a 4,800-image field set](https://etasr.com/index.php/ETASR/article/download/12646/5804); [fruit disease study using ICAR Hirehalli and Kaggle data](https://arccjournals.com/journal/indian-journal-of-agricultural-research/A-6404) | 8,847 reported for nine classes in the Kaggle dataset; about 300 in the public GitHub dataset; 4,800 in a separate paper-described field set; 1,115 fruit images in another study | The 8,847-image dataset's direct download URL and licence need verification; the 4,800-image paper does not itself establish a public download or licence; verify terms before use | Promising additional leads. Do not assume these counts can be added together or that the sources are downloadable/licensed. Check overlap and source rights first. |
| Cardamom | [Cardamom Dataset 2021, GitHub](https://github.com/sunilchinnahalli/Cardamom-Dataset-2021); [2026 cardamom leaf severity study](https://openurl.ebsco.com/contentitem/gcd%3A196720609?crl=f&id=ebsco%3Agcd%3A196720609&jrnl=22414487&sid=ebsco%3Aplink%3Acrawler-gcd); [Indian Biological Images Archive spice-recognition dataset](https://ibdc.dbt.gov.in/ibia/study_details_browse_l/PPS_1000000011/) | 1,724 cardamom disease/healthy leaf images in the 2021 dataset; a separate 2026 study reports 1,444 field leaf images; IBDCA dataset contains 10,991 images across 19 spices, including green cardamom and black cardamom as dried spice products | The 1,724-image dataset says available upon request and terms are unclear; obtain permission for the 1,444-image study; IBDCA spice photos are not disease-leaf photos and are **not suitable** as labelled disease training images | The first two are disease-data leads requiring access/rights checks. Spice-recognition data can only support a separate species/product recognition task, not disease classification. |
| Arecanut | [Arecanut disease research using a proprietary 1,100-photo dataset](https://ijeecs.iaescore.com/index.php/IJEECS/article/view/32456) and [a separate arecanut classification study](https://link.springer.com/chapter/10.1007/978-3-032-19318-6_20) | 1,100 proprietary photos; 1,578 leaf images reported in another study | Proprietary/custom collection; no open download established from these papers | Contact authors for permission/access; do not count as open data until rights and files are confirmed. |

## Arecanut: recommended acquisition order

1. **Investigate the reported 8,847-image Kaggle dataset first.** A 2026 paper lists class counts: Healthy Leaf 604, Healthy Nut 1,886, Healthy Trunk 1,432, Mahali Koleroga 2,563, Stem Bleeding 151, Bud Borer 144, Healthy Foot 50, Stem Cracking 540, Yellow Leaf Disease 1,477. This totals 8,847, but the paper's existence does not guarantee the original dataset is openly downloadable or reusable. Find the cited Kaggle source, confirm licence and version, then inspect image files.
2. Check the separate 4,800-image field study and request a data link/permission from its authors if no public repository is listed.
3. Download the ~300-image GitHub dataset only after confirming its licence; it may overlap with papers/datasets from the same authors.
4. Ask authors of the 1,100- and 1,578-image research datasets whether the original images and reuse rights are available.
5. Merge only after hash/perceptual deduplication, class-name reconciliation, and provenance tracking. These reported totals are **not additive** until overlap and rights are checked.

## Cardamom: recommended acquisition order

1. Request access and licence terms for the 1,724-image [Cardamom Dataset 2021](https://github.com/sunilchinnahalli/Cardamom-Dataset-2021). The paper describes three leaf classes: Colletotrichum blight (280), healthy (781), and Phyllosticta leaf spot (663).
2. Contact the authors of the 2026 study reporting 1,444 field-collected cardamom leaf images; ask for the dataset, class counts, labels, and written reuse terms. Its paper abstract alone is not a public data licence.
3. Search institutional repositories and contact the Indian Cardamom Research Institute / Spices Board research contacts for permitted research image access. Any newly found collection must be checked for rights and annotation reliability.
4. Do **not** use the 10,991-image Indian spice-recognition dataset to inflate cardamom disease training counts: its green-cardamom images are seed/seedpod product photographs and the dataset is for spice recognition, not plant disease.
5. If open disease images remain below 5,000, organise field collection in cardamom-growing regions with consent and expert labelling. Keep healthy, each disease, severity stage, plant part, location/session, and source identity in the manifest.

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

The public sources found so far provide promising crop-level coverage for Coconut, Banana, and Paddy/Rice. Black Pepper has a candidate source whose downloadable version/count must be checked. Arecanut has a potentially useful 8,847-image dataset lead, but direct source/licence verification is still required. Cardamom remains data-limited: one 1,724-image collection plus a reported 1,444-image study need access and licence checks. It is **not yet accurate to claim that all six crops have 5,000–10,000 clean, licensed images ready for final training**. Final training should begin only after the manifest, licences, class taxonomy, and leakage checks are reviewed.
