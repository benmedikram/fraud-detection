# Credit Card Fraud Detection

An end-to-end machine learning project that detects fraudulent credit card transactions, from data cleaning to an API deployed with Docker.

## Problem

Card fraud costs banks and customers billions every year, and fraudulent transactions are extremely rare (about 0.17% of all transactions). The goal is to flag suspicious transactions automatically while blocking as few legitimate customers as possible.

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (ULB Machine Learning Group): 284,807 European card transactions over two days, 492 of them fraud. Features V1-V28 are anonymized (PCA); Time, Amount and Class are not.

## Progress

- [x] Step 1: data cleaning, exploration, splits, baseline
- [x] Step 2: models and imbalance handling
- [x] Step 3: tuning, threshold, explainability
- [x] Step 4: API, Docker, tests, CI

## Data findings

- Removed 1081 duplicate rows, leaving 283726 transactions (473 frauds).
- Stratified 70/15/15 split: train 198608, validation 42559, test 42559 rows. The fraud rate is about 0.17% in each.
- Amount is heavily skewed, so I added a log-transformed version. Fraud amounts are not clearly higher than normal ones.
- V17, V14, V12 and V10 are the features that differ most between fraud and normal transactions.
- A "never fraud" baseline gets 99.833% accuracy but 0% recall and a PR-AUC of about 0.0017, so accuracy is not a useful metric here. I use PR-AUC and recall.

## How to reproduce

1. Download creditcard.csv from Kaggle and put it in data/raw/
2. Install the requirements: python -m pip install -r requirements.txt
3. Prepare the data: python -m src.data
4. Run the tests: python -m pytest

## Model Comparaison (5-fold CV on the training set)

| Model | Imbalance strategy | PR-AUC (mean ± std) | Recall @0.5 | Precision @0.5 |
|---|---|---|---|---|
| XGBoost | scale_pos_weight sqrt | 0.855 ± 0.028 | 0.825 | 0.918 |
| XGBoost | none | 0.855 ± 0.027 | 0.801 | 0.944 |
| XGBoost | scale_pos_weight full | 0.855 ± 0.029 | 0.828 | 0.896 |
| RandomForest | SMOTE | 0.846 ± 0.027 | 0.831 | 0.863 |
| RandomForest | none | 0.843 ± 0.024 | 0.764 | 0.935 |

Best model: **XGBoost**, PR-AUC ~0.855, far above the 0.0017 "never fraud" baseline.
Accuracy isn't used as a metric, since a model that never flags fraud already scores 99.8%.
All 11 experiments were tracked with MLflow.

![MLflow runs table, sorted by PR-AUC](docs/mlflow_runs_table.png)

![Parallel coordinates: imbalance strategy, model, and PR-AUC](docs/mlflow_parallel_coords.png)

![XGBoost best run detail](docs/mlflow_xgboost_detail.png)

## Tuning, Threshold, and Explainability

**Tuning (Optuna, 50 trials, 5-fold CV):**
Best CV PR-AUC: 0.859 (vs 0.847 for the manually-tuned model).
Best hyperparameters: `n_estimators=401`, `max_depth=5`, `learning_rate=0.092`, `subsample=0.76`, `colsample_bytree=0.91`, `min_child_weight=7` `gamma=1.52`, `scale_pos_weight=169.9`. Most influential hyperparameters: `colsample_bytree`, `scale_pos_weight`, `subsample`.

**Threshold (cost-based selection):**
Assumed costs: a missed fraud costs $100, a false alarm costs $5. Sweeping
thresholds on the validation set found the cost-minimizing threshold at **0.05**
(well below the default 0.5, since missed frauds are far more costly than false alarms). This reduced estimated cost from $1,615 (default threshold) to $1,190, a **26.3% savings**.

**Explainability (SHAP):**
The model relies most heavily on V4, followed by V14, V12, V10, and V3 — largely consistent with earlier correlation analysis, though SHAP surfaced V4 as more central than simple correlation suggested, since it captures feature interactions. 
A reviewed "missed fraud" case had a predicted probability of just 0.001 — every major feature looked statistically like a normal transaction, showing a genuine limit of what these anonymized features alone can catch.

**Robustness (time-based split):**
Training on the first 80% of transactions chronologically and testing on the last 20% (rather than a random split) checks for concept drift within the 2-day window. PR-AUC dropped from **0.85** (random split) to **0.79** (time-based split), indicating the model does not generalize perfectly across time even within this short window. In production, this would call for regular retraining and drift monitoring — though 2 days is too short a window to draw strong conclusions about long-term drift patterns; a longer historical dataset would be needed to properly characterize this.

## Final Test Results

Evaluated once on the held-out test set, at the cost-optimized threshold (0.05), never used for model selection or tuning:

| Metric | Value |
|---|---|
| PR-AUC | 0.822 |
| Precision | 0.619 |
| Recall | 0.845 |
| Frauds caught | 60 / 71 |
| False alarms | 37 |
| Estimated cost savings vs default threshold | 26.3%* |

*Cost savings measured on the validation set during threshold selection; the same threshold was then applied once to the test set above.

**Interpretation:** at this low threshold, the model catches 84.5% of frauds at the cost of flagging 37 legitimate transactions per ~71 frauds. This reflects a deliberate business trade-off (missed fraud costs 20x more than a false alarm in our cost model) — a bank could review these 37 flagged transactions manually at a much lower cost than the frauds they'd otherwise miss.

## Deployment

**Run everything with Docker Compose:**
```bash
docker compose up --build
```
- API: http://localhost:8000/docs
- Demo: http://localhost:8501

**Run just the API:**
```bash
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api
```

**API example:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0, -1.36, -0.07, 2.54, ...]}'
```

## Tech Stack

Python · pandas · scikit-learn · XGBoost · Optuna · SHAP · MLflow · FastAPI ·
Streamlit · Docker · pytest · GitHub Actions

## Project Structure

```
fraud-detection/
├── src/            # data pipeline, model wrapper, FastAPI app
├── app/            # Streamlit demo
├── notebooks/      # EDA, modeling, tuning (with full analysis)
├── models/         # trained model, threshold, metadata
├── tests/          # pytest suite
├── docs/           # screenshots for this README
└── docker-compose.yml
```