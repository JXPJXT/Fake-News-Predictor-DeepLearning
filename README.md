# Fake News Predictor  
**AI-Powered Misinformation Detection System**

<div align="center">

![Project Status](https://img.shields.io/badge/Status-Active-success?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square)
![Framework](https://img.shields.io/badge/FastAPI-0.95+-005571?style=flat-square)
![ML](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat-square)
![Deployment](https://img.shields.io/badge/AWS-EC2%20%7C%20Kubernetes-orange?style=flat-square)

**A production-ready deep learning system for classifying news articles as Real or Fake, achieving ~96% accuracy on a large-scale benchmark dataset.**

</div>

---

## 🚀 Overview

In today's information landscape, misinformation spreads rapidly and undermines trust. This project delivers a robust, scalable **fake news detection** pipeline that combines powerful contextual embeddings from **BERT** (via Hugging Face Transformers) with sequential modeling via **LSTM** to classify full news articles with high reliability.

Built with enterprise-grade practices in mind:

- Large-scale data processing on **Databricks**
- Automated ETL workflows via **Apache Airflow**
- High-performance model serving with **FastAPI**
- Containerized & orchestrated deployment on **AWS** (EC2 + Kubernetes)

The system has been evaluated on the well-known **WELFake** dataset — a challenging, merged corpus of ~72,134 articles (≈35k real + ≈37k fake) sourced from Kaggle, McIntire, Reuters, BuzzFeed Political, and others.

---

## 📊 Performance Highlights

| Metric          | Value          | Notes                              |
|-----------------|----------------|------------------------------------|
| **Accuracy**    | **95.96%**     | Test set (WELFake)                 |
| **Precision**   | ~96%           | (depending on class balance)       |
| **Recall**      | ~96%           |                                    |
| **F1-Score**    | ~96%           | Balanced metric                    |
| **Dataset Size**| 72,134 articles| 35,028 real • 37,106 fake          |
| **Inference**   | Real-time      | < 500 ms per article (avg.)        |

> Note: State-of-the-art research models on WELFake now reach 98–99% with more advanced ensembles / fine-tuning. Our hybrid BERT+LSTM offers a strong, reproducible baseline suitable for production use.

---

## 🛠 Tech Stack

| Layer              | Technologies                                                                 |
|--------------------|------------------------------------------------------------------------------|
| **ML / Modeling**  | Python • TensorFlow/Keras • Hugging Face Transformers (BERT) • Scikit-learn |
| **Data Processing**| Pandas • NumPy • NLTK • Databricks (Spark)                                  |
| **ETL / Workflow** | Apache Airflow                                                               |
| **API / Serving**  | FastAPI • Uvicorn                                                            |
| **Deployment**     | Docker • Kubernetes • AWS EC2                                                |
| **Frontend**       | Static HTML + JavaScript + CSS (simple UI)                                   |

---

## 🏗 System Architecture

1. **Data Pipeline**  
   - **Source**: WELFake (IEEE TCSS 2021) – merged, deduplicated, cleaned corpus  
   - **Processing**: Distributed cleaning & feature extraction on Databricks  
   - **Orchestration**: Airflow DAGs for reproducible ETL (ingest → clean → split → embed)

2. **Model Design** — Hybrid BERT + LSTM Classifier  
   - Pre-trained **BERT** generates rich, context-aware token embeddings  
   - **LSTM** layer(s) model long-range dependencies across the article  
   - Classification head + Dropout + Early Stopping + AdamW optimizer  
   - Goal: balance strong contextual understanding with sequential pattern detection

3. **Serving & Deployment**  
   - Model wrapped as REST API via **FastAPI** (OpenAPI docs included)  
   - Dockerized for portability  
   - Kubernetes on AWS EC2 for horizontal scaling & high availability

---

## ⚡ Key Features

- **Real-time classification** — POST title + text or full article  
- **URL ingestion** (optional) — fetch & classify live articles  
- **Explainability hooks** — attention visualization / important token highlighting (future)  
- **Article summarization** — optional lightweight generative summary + confidence score  
- **Handles noisy & diverse text** — robust to headlines, clickbait, varying lengths

---

## 🔧 Quick Start (Local Development)

### Prerequisites
- Python 3.9+
- Docker (recommended for model serving)
- ~16 GB RAM + GPU strongly recommended for training

### Steps

1. Clone & enter directory
   ```bash
   git clone https://github.com/yourusername/fake-news-predictor.git
   cd fake-news-predictor
