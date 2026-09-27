\# Scalable Entity Resolution using XGBoost



A scalable machine learning pipeline for entity resolution and record linkage across large business datasets.



\## Overview



The project matches entities from Source 1 against entities from Source 2 and Source 3 despite variations in:



\- Business names

\- Addresses

\- Country information

\- Spelling and typographical errors

\- Word ordering

\- Missing or additional information



The pipeline combines multi-pass candidate generation, fuzzy string similarity, feature engineering and XGBoost classification.



\## Dataset Scale



\### Training



\- Source 1: 2,206,821 records

\- Source 2: 5,034,616 records

\- Source 3: 5,285,603 records

\- Target pool: 10,320,219 entities



\### Test



\- Source 1: 1,732,544 records

\- Source 2: 4,887,273 records

\- Source 3: 5,082,316 records

\- Target pool: 9,969,589 entities



\## Approach



\### 1. Data Cleaning



Records are normalized using:



\- Lowercasing

\- Punctuation normalization

\- Whitespace normalization

\- Corporate abbreviation normalization

\- Address abbreviation normalization

\- Missing-value handling



\### 2. Candidate Generation



A multi-pass candidate-generation strategy reduces the search space before machine learning.



The final TEST candidate set contained:



\- 1,732,544 Source 1 entities

\- 51,810,001 candidate IDs



This represents approximately a 99.7% reduction compared with brute-force comparison against the complete target pool.



\### 3. Feature Engineering



The model uses 11 features:



\- Name ratio

\- Name token sort similarity

\- Name token set similarity

\- Name partial similarity

\- Address ratio

\- Address token set similarity

\- Address partial similarity

\- Country match

\- Missing address indicator

\- Name length difference

\- Address length difference



\### 4. Machine Learning



An XGBoost classifier is used to predict whether a candidate pair represents the same entity.



\## Validation Results



The final source-level held-out validation produced:



| Metric | Result |

|---|---:|

| Precision | 99.19% |

| Recall | 89.14% |

| F0.5 | 0.9700 |



The best tuned model achieved PR-AUC of approximately 0.997.



\## Final Test Inference



The final inference processed:



\- 1,732,544 Source 1 entities

\- 48,594,924 candidate pairs scored

\- 50,283,577 predicted links



Structural validation produced:



\- Invalid target IDs: 0

\- Duplicate target IDs: 0

\- Duplicate Source 1 IDs: 0



\## Output



The challenge submission consists of:



\- `matching\_results.tsv`

\- `candidate\_pairs.tsv`



These large files are intentionally excluded from GitHub.



\## Model



The trained XGBoost model is stored locally as:



`output/xgb\_best\_v2.json`



The selected inference threshold is:



`output/best\_threshold\_v2.txt`



\## Reproducibility



Large datasets and generated submission files are excluded from this repository because of their size.



Place the required datasets in the expected local directories and install dependencies with:



```bash

pip install -r requirements.txt

