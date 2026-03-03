# AG News Topic Classification using Transformers

This project focuses on categorizing news articles into four distinct categories—**World, Sports, Business, and Sci/Tech**—using advanced Natural Language Processing (NLP) techniques. The project evaluates several transformer-based models, including **BERT**, **DistilBERT**, and **ALBERT**, to determine the most effective approach for text classification.

## 🚀 Project Overview
The primary goal is to leverage pre-trained transformer models to accurately classify articles from the **AG's News Topic Classification Dataset**, which consists of 120,000 samples.

### Models Implemented

**Model 1-4: DistilBERT**: A lightweight, fast, and efficient version of BERT that retains most of its performance.
**Model 5: BERT**: A sophisticated, high-performance model developed by Google.
**ALBERT (Trials)**: Tested for its compact and computationally efficient architecture.


## 🛠️ Technical Implementation

### Data Preprocessing
To ensure high-quality input for the models, a comprehensive cleaning pipeline was implemented:

**Text Cleaning**: Lowercasing, removing punctuation, HTML tags, numbers, and URLs.
**Tokenization**: Converting text into words and subsequently into model-specific tokens using `DistilBertTokenizer`.
**Stopword Removal & Stemming**: Reducing noise by removing common words and reducing others to their base forms using the Porter Stemmer.
**Label Encoding**: Converting categorical class labels into numeric format using `LabelEncoder` and one-hot encoding.

### Training Configuration

**Optimizer**: Adam and AdamW optimizers were used with varying learning rates (e.g., $2\times10^{-5}$ to $5\times10^{-5}$).
**Loss Function**: Categorical Crossentropy (from logits).
**Environment**: Models were trained on **Google Colab** to utilize cloud-based computational resources.


## 📊 Results and Comparison
The models were evaluated based on **Accuracy**, **Confusion Matrix**, and **Classification Reports** (Precision, Recall, and F1-Score).

