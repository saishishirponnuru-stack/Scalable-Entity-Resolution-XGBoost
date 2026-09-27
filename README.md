# Scalable Entity Resolution with XGBoost

A scalable machine learning pipeline for resolving and matching business entities across large, noisy datasets using multi-pass candidate generation, fuzzy matching, feature engineering, and XGBoost classification.

## 🚀 Overview

Entity resolution is the process of identifying records that refer to the same real-world entity, even when business names, addresses, or other attributes contain spelling variations, abbreviations, formatting differences, or missing information.

This project uses a **Candidate Generation → Feature Engineering → XGBoost → Matching** pipeline to efficiently perform entity resolution at large scale.

## 🔄 Pipeline

Raw Data
   ↓
Data Cleaning & Normalization
   ↓
Multi-Pass Candidate Generation
   ↓
Similarity Feature Engineering
   ↓
XGBoost Classification
   ↓
Threshold-Based Matching
   ↓
Validated Output
📊 Scale

The pipeline was developed for millions of business records:

Dataset	Records
TRAIN Source 1	2.20M
TRAIN Source 2	5.03M
TRAIN Source 3	5.29M
TEST Source 1	1.73M
TEST Source 2	4.89M
TEST Source 3	5.08M

A brute-force comparison across the TEST data would involve approximately 17.27 trillion potential comparisons.

Candidate generation reduced this to approximately 51.81 million candidate pairs.

🧠 Machine Learning

The model uses 11 engineered features based on:

Business name similarity
Token-based name similarity
Partial name similarity
Address similarity
Token-based address similarity
Country matching
Missing-value indicators
Name and address length differences

An XGBoost binary classifier is then used to distinguish between matching and non-matching entity pairs.

Model
XGBoost
500 estimators
Learning rate: 0.08
Maximum depth: 7
Histogram-based training
Regularization
Source-level validation split
📈 Validation Results

The comprehensive held-out validation produced:

Metric	Result
Precision	99.19%
Recall	89.14%
F0.5 Score	0.9700

An earlier model-tuning experiment achieved a maximum validation F0.5 of 0.9776 under a different validation setup.

These are offline validation results. The official hidden TEST leaderboard score is not available locally.

🧪 Final TEST Inference

The final pipeline processed 1,732,544 TEST Source 1 entities and scored approximately 48.59 million candidate pairs.

Final output:

50,283,577 predicted entity links
1,732,503 Source 1 entities with matches
41 Source 1 entities without a predicted match
0 invalid target IDs
0 duplicate Source 1 IDs
🛠️ Tech Stack
Python
Pandas
NumPy
RapidFuzz
Scikit-learn
XGBoost


📁 Project Structure
Scalable-Entity-Resolution-XGBoost/
│
├── CODE/
│   ├── Scrub.py
│   ├── candidate_genaration.py
│   ├── training_data_set_creation.py
│   ├── XGBoost.py
│   ├── final_inference.py
│   └── validation scripts
│
├── output/
│   ├── xgb_best_v2.json
│   └── best_threshold_v2.txt
│
├── requirements.txt
├── .gitignore
└── README.md

Large datasets and generated TSV files are intentionally excluded from the repository.

🎯 Key Highlights
Designed for large-scale entity resolution
Multi-pass candidate retrieval instead of brute-force matching
Fuzzy and token-based similarity features
XGBoost-based classification
Handles noisy business names and addresses
Scalable processing across millions of records
Output validation for duplicate and invalid entity IDs
