# Amazon Food Reviews Sentiment Analysis

## Overview

This project classifies Amazon food reviews into negative, neutral, and positive sentiment using deep learning models for natural language processing.

The project is designed as an undergraduate AI/NLP portfolio project. It includes a cleaned experiment notebook, reusable training scripts, dataset setup notes, and a short results report.

## Dataset

The project uses the Amazon Fine Food Reviews dataset. The raw dataset files are not included in this repository because they are large.

Expected local file:

```text
Reviews.csv
```

Place `Reviews.csv` in the project root before running the notebook or scripts. See [data/README.md](data/README.md) for dataset details.

## Models Used

- LSTM
- Simple RNN

## Preprocessing

- Removed missing review text and score values
- Filtered reviews by word count
- Converted review scores into sentiment labels
- Cleaned text with regular expressions
- Removed English stopwords
- Tokenized text
- Padded sequences to a fixed length
- Encoded sentiment labels for training

## Project Structure

```text
amazon-food-reviews-sentiment/
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- LICENSE
|-- notebooks/
|   `-- sentiment_analysis_lstm_rnn.ipynb
|-- src/
|   |-- preprocessing.py
|   |-- train_lstm.py
|   |-- train_rnn.py
|   `-- evaluate.py
|-- reports/
|   `-- results.md
`-- data/
    `-- README.md
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook notebooks/sentiment_analysis_lstm_rnn.ipynb
```

Train from scripts:

```bash
python src/train_lstm.py --data Reviews.csv
python src/train_rnn.py --data Reviews.csv
```

Quick smoke tests:

```bash
python src/preprocessing.py --sample-size 100
python src/train_rnn.py --sample-size 100 --epochs 1 --batch-size 16
```

## Results

Results are summarized in [reports/results.md](reports/results.md). Re-run the notebook or training scripts after downloading the dataset to regenerate final metrics.

## Future Improvements

- Add a Bidirectional LSTM model
- Compare with classical machine learning models such as Logistic Regression and Naive Bayes
- Use pretrained word embeddings such as GloVe
- Handle class imbalance with sampling or class weights
- Add plots for training and validation accuracy/loss
- Save trained models and tokenizers for inference
