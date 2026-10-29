# Open dataset catalog for the six target crops

**Important:** the entries below are discovery leads, not an assertion that all data are suitable for every use. Re-check the source's current license, paper, class definitions, and download terms before inclusion. Preserve attribution and citations. Do not mix classes without an explicit reviewed mapping.

## Direct target-crop sources

| Crop | Dataset | Reported scale / classes | Source and decision notes |
|---|---|---|---|
| Coconut | Coconut Tree Disease Dataset | 5,798 images; five disease classes: bud root dropping, bud rot, gray leaf spot, leaf rot, stem bleeding | Mendeley Data DOI: https://data.mendeley.com/datasets/gh56wbsnj5/1. Published data article describes field images collected in Maharashtra. Confirm license on the dataset record and cite the data article. |
| Coconut | Coconut Tree Disease Classification repository | 5,798 images; five disease classes; repository states CC BY 4.0 | https://github.com/ai-agriculture-circuits-and-systems/coconut_tree_disease_classification. Repository says it hosts structure/conversion scripts and that original images must be downloaded separately; verify original source and attribution. |
| Arecanut | Arecanut Disease Detection Dataset | About 300 field RGB images; six folders/classes including healthy leaf/nut, bud rot, koleroga, ring spot, yellow leaf | https://github.com/BeenaPradeep/ARECANUT-DATASET. Small sample size; inspect license file and class definitions. Treat healthy leaf and healthy nut as potentially different task labels, not interchangeable. |
| Paddy/rice | Paddy Disease Classification | Repository reports 10,407 images and 10 categories | https://github.com/ai-agriculture-circuits-and-systems/paddy_disease_classification and original Kaggle competition https://www.kaggle.com/competitions/paddy-disease-classification. Check original terms and avoid duplicate images across splits. |
| Paddy/rice | Rice Leaf Bacterial and Fungal Disease Dataset | 1,701 original images; eight rice leaf categories; source also includes augmented images | https://data.mendeley.com/datasets/hx6f852hw4/2. Reported CC BY 4.0. Prefer original images for independent splitting; augmented copies must remain in the same split as their source. |
| Banana | Banana Leaves Imagery Dataset | 11,767 images: 3,339 healthy, 3,496 Black Sigatoka, 4,932 Fusarium Wilt Race 1 | Open-access data descriptor: https://www.nature.com/articles/s41597-025-04456-4. Collected in Tanzania; evaluate regional/domain shift before applying to South Asian farms. Read the article's data availability and license terms. |
| Banana | PSFD-Musa dataset | Indian Biological Images Archive lists a dataset of banana plant, stem, fruit, leaf, and disease imagery | https://ibdc.dbt.gov.in/ibia/studybrowse/. Open the study record to inspect the actual disease labels, metadata, license, and download procedure before inclusion. |

## Cross-crop / pretraining and discovery sources

| Source | Scale / value | Caution |
|---|---|---|
| PlantVillage | Common benchmark for leaf disease representation learning | Many images have clean backgrounds; not a replacement for field images. Check current license and crop coverage: https://github.com/spMohanty/PlantVillage-Dataset |
| PlantDoc | Field-style images with varied backgrounds | Check class coverage and license: https://github.com/pratikkayal/PlantDoc-Dataset |
| Agriculture Collections index | Curated index of many agricultural vision datasets; useful for discovering further crop-specific sources | Treat it as an index, not a license grant. Follow each original source: https://github.com/ai-agriculture-circuits-and-systems/agriculture-collections |
| Mendeley Data | Searchable research data with DOI and license metadata | Search each crop name and disease name; assess individual record terms: https://data.mendeley.com |
| Zenodo | Research datasets with DOI and license metadata | Review each record's actual license and paper: https://zenodo.org |
| Hugging Face Datasets | Searchable dataset hub and dataset cards | Validate the original provenance and license, not just the hub listing: https://huggingface.co/datasets |
| Kaggle Datasets and competitions | Broad discovery source, including paddy disease challenge data | Terms vary by dataset/competition; use official download mechanisms and follow competition rules: https://www.kaggle.com/datasets |

## Missing or weakly covered crops

Black pepper and cardamom need a dedicated targeted search and likely new expert-verified field collection. The sources above do not establish adequate labeled coverage for either crop. Arecanut has a promising but small field set; coconut and paddy have more promising image counts, but still require independent provenance and leakage checks. Do not fill gaps with synthetic images and treat them as real observations.

## Dataset intake checklist

- [ ] Save the canonical source URL, paper/DOI, access date, author/citation, license text or license URL, and permitted-use decision.
- [ ] Download using the source's official permitted mechanism; record archive checksum and original filename.
- [ ] Keep raw files immutable and never commit image archives or credentials to Git.
- [ ] Inspect labels and sample images; record crop, plant part, class, severity, source region, capture setting, and annotation confidence.
- [ ] Exclude sources with unclear licensing until permission is resolved.
- [ ] Identify duplicates across sources using file hashes, perceptual hashes, and manual review of likely matches.
- [ ] Group splits by plant/farm/session/source lineage. Never split augmented siblings across train, validation, and test.
- [ ] Use a locked independent test set from farms/regions or time periods not represented in training.
- [ ] Have a plant pathologist or qualified agronomist review class definitions and uncertain samples.
- [ ] Recompute class counts and manifest checksum after every approved data change.
