# Fake News Predictor: AI-Powered Misinformation Detection

<div align="center">

![Project Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Framework](https://img.shields.io/badge/FastAPI-0.95%2B-005571)
![ML](https://img.shields.io/badge/Hugging%20Face-BERT-yellow)
![Cloud](https://img.shields.io/badge/AWS-EC2%20%7C%20Kubernetes-orange)

**An enterprise-grade Deep Learning system engineered to detect fake news with 95.96% accuracy.**

</div>

---

## 🚀 Overview

In an era of rampant misinformation, this project leverages state-of-the-art **Natural Language Processing (NLP)** to classify news articles as **Real** or **Fake**. By combining the sequential modeling power of **LSTM** with the contextual understanding of **BERT (Hugging Face)**, we have achieved industry-leading performance on the massive **WELFake** dataset.

The system is designed for scale, utilizing **Databricks** for big data processing, **Apache Airflow** for automated ETL pipelines, and deployed via **FastAPI** on **AWS EC2** with **Kubernetes** orchestration.

---

## 📊 Key Metrics

- **Accuracy**: **95.96%**
- **Dataset**: **72,000+** articles (WELFake Dataset)
- **Model**: Hybrid **LSTM + BERT** Classifier
- **Latency**: Real-time inference capability

---

## 🛠 Tech Stack

| Domain | Technologies |
| :--- | :--- |
| **Machine Learning** | Python, Scikit-Learn, TensorFlow/Keras, Hugging Face Transformers (BERT) |
| **Big Data & ETL** | Databricks, Apache Airflow |
| **Backend API** | FastAPI, Uvicorn |
| **Deployment** | AWS EC2, Kubernetes (K8s), Docker |
| **Frontend** | JavaScript, HTML5, CSS3 |
| **Data Processing** | Pandas, NumPy, NLTK |

---

## 🏗 Architecture & Methodology

### 1. Data Pipeline (ETL)
- **Source**: The **WELFake** dataset (IEEE TCSS, 2021), a merger of Kaggle, McIntire, Reuters, and BuzzFeed Political datasets.
- **Processing**: Utilized **Databricks** for distributed processing of the 72k+ articles.
- **Orchestration**: **Apache Airflow** manages the extract, transform, and load (ETL) workflows to ensure data consistency and reproducibility.

### 2. Model Architecture
We engineered a hybrid Deep Learning model to capture both long-term dependencies and deep contextual meaning:
- **BERT (Bidirectional Encoder Representations from Transformers)**: from Hugging Face is used to generate rich, contextualized word embeddings.
- **LSTM (Long Short-Term Memory)**: Processes the sequences of these embeddings to identify patterns over article length.
- **Optimization**: Tuned using Adam optimizer, with techniques like Dropout and EarlyStopping to prevent overfitting.

### 3. Deployment Strategy
- **Microservice**: The model is served as a RESTful API using **FastAPI** for high performance and auto-generated documentation.
- **Containerization**: Application is containerized using Docker.
- **Orchestration**: Deployed on **AWS EC2** instances managed by **Kubernetes** for scaling and high availability.

---

## ⚡ Features

- **Real-time Prediction**: Instant classification of input text or URLs.
- **Live News Aggregation**: Fetches and analyzes current news topics using external news APIs.
- **Summarization**: Integrated Generative AI to provide concise summaries of long articles alongside reliability scores.
- **Robustness**: Handles diverse writing styles and noisy data effectively.

---

## 🔧 Installation & Setup

### Prerequisites
- Python 3.9+
- Docker & Kubernetes (for deployment)
- AWS Account (optional for cloud deployment)

### Local Development
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/fake-news-predictor.git
   cd fake-news-predictor
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the API**
   ```bash
   uvicorn app:app --reload
   ```

4. **Access the Interface**
   - API Docs: `http://localhost:8000/docs`
   - Frontend: Open `Frontend/index.html` in your browser.

---

## 👥 Contributors

- **Bhatia** (Lead Engineer)
- **Navya** & **Ananjay** (Contributors)

---

<div align="center">
<i>Built with ❤️ for a truthful internet.</i>
</div>
