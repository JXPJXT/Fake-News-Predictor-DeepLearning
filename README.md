# Fake News Predictor

## Overview
This project classifies news articles as real or fake using AI. With **Natural Language Processing (NLP)** and **deep learning**, we’re fighting misinformation head-on.

## Dataset 
- **WELFake dataset** (IEEE TCSS, 2021): 72,134 articles (35,028 real, 37,106 fake).  
- **Source**: Merged Kaggle, McIntire, Reuters, and BuzzFeed Political datasets.  
- **Structure**: Serial No., Title, Text, Label (0 = Fake, 1 = Real).  

## Methodology
- **Preprocessing**: Text cleaning, tokenization, lemmatization, stopword removal.  
- **Big Data Analysis**: **Apache Spark** for title pattern analysis.  
- **Model**: **LSTM** with **Word2Vec** embeddings.  
- **Training**: 10 epochs, EarlyStopping, multi-GPU.  

## Results
- **Accuracy**: 95.96%  
- **Precision**: 0.95 (Fake), 0.97 (Real)  
- **Recall**: 0.97 (Fake), 0.95 (Real)  
- **F1-Score**: 0.96 (both)  

## Deployment
- **Backend**: **Flask API** on **AWS EC2**.  
- **Frontend**: HTML/CSS/JS on **AWS S3**.  
- **Features**: Real-time predictions, live news fetching.  

## Challenges & Solutions
- **Overfitting**: Dropout, EarlyStopping.  
- **Slow Preprocessing**: Multiprocessing.  
- **API Issues**: Error handling, NewsAPI fallback.  

## Future Work
- Real-time scraping.  
- Transformers (BERT/RoBERTa).  
- Full-stack upgrade.  
- Multilingual support.  

## Credits
A special thanks to Navya and Ananjay for their contributions, and to our mentors for their guidance!
