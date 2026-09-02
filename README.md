# Phishing Email Detector

A machine learning pipeline that classifies emails as phishing or legitimate, 
comparing classical ML baselines against a fine-tuned transformer, deployed 
as a containerized REST API.

## Overview

This project explores four modeling approaches for binary phishing/legitimate 
email classification, evaluates them honestly against each other, and deploys 
the most practical model as a production-style API.

## Dataset

[Phishing Email Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset) 
— 82,486 emails combining six source corpora (Enron, Ling, CEAS, Nazario, 
Nigerian Fraud, SpamAssassin). Text was pre-processed (lowercased, punctuation 
and stopwords removed) prior to publication.

## Models compared

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Naive Bayes (TF-IDF) | 96.17% | 97.86% | 94.71% | 96.26% |
| Logistic Regression (TF-IDF) | 98.10% | 97.96% | 98.40% | 98.18% |
| Linear SVM (TF-IDF) | 98.51% | 98.65% | 98.50% | 98.57% |
| Fine-tuned DistilBERT | 99.10% | 99.20% | 99.00% | 99.09% |

Recall was prioritized as the key metric, since a missed phishing email 
(false negative) carries more real-world risk than a false alarm.

## Key decisions and limitations

- **DistilBERT was fine-tuned on a stratified 30,000-row subset** (15,000 per 
  class) rather than the full dataset, due to time constraints. This means 
  the transformer comparison is not perfectly controlled for data volume — 
  results should be read as a proof-of-concept, not a definitive verdict on 
  transformer vs. classical performance for this task.
- Training was limited to 2 epochs, as training loss showed early signs of 
  overfitting relative to validation loss by epoch 2.
- **Linear SVM was chosen for deployment** over the more accurate DistilBERT, 
  prioritizing inference speed, memory footprint, and deployment simplicity — 
  DistilBERT's ~0.6 percentage point accuracy gain did not justify a 
  significantly larger, slower-to-serve model for this application.
- Tree-based models (Random Forest, XGBoost) were not evaluated, as they are 
  generally less effective on high-dimensional sparse TF-IDF representations 
  than linear models; they would be stronger candidates on engineered 
  numerical features or with dimensionality reduction applied first.

## Project structure

    phishing-email-detector/
    ├── src/           # preprocessing, training, evaluation
    ├── app/           # FastAPI serving app + Dockerfile
    ├── models/        # saved model artifacts (gitignored)
    ├── notebooks/      # exploration and DistilBERT fine-tuning
    └── data/          # dataset (gitignored)

## Running locally

    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    python src/train.py
    uvicorn app.main:app --reload

## Running with Docker

    docker build -t phishing-detector -f app/Dockerfile .
    docker run -p 8000:8000 phishing-detector

Then visit `http://127.0.0.1:8000/docs` for the interactive API.

## Example request

    curl -X POST 'http://127.0.0.1:8000/predict' \
      -H 'Content-Type: application/json' \
      -d '{"text": "urgent click link verify account now"}'

    → {"prediction": "phishing", "label": 1}

## Future work

- Fine-tune DistilBERT on the full 82,486-row dataset given more compute time
- Add prediction confidence scores via `CalibratedClassifierCV`
- Explore engineered features (email length, link count) for tree-based models
