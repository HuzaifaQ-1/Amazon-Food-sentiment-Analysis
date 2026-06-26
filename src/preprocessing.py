import re
import argparse

import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split


LABEL_MAP = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

REVERSE_LABEL_MAP = {value: key for key, value in LABEL_MAP.items()}


def get_sentiment(score):
    if score >= 4:
        return "positive"
    if score == 3:
        return "neutral"
    return "negative"


def load_stopwords():
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords")
        return set(stopwords.words("english"))


def clean_text(text, stop_words=None):
    if stop_words is None:
        stop_words = load_stopwords()

    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)


def load_and_clean_reviews(
    csv_path,
    sample_size=10000,
    random_state=42,
    min_words=50,
    max_words=500,
):
    df = pd.read_csv(csv_path)

    if sample_size:
        df = df.sample(sample_size, random_state=random_state)

    df = df[["Text", "Score"]].dropna()
    df["word_count"] = df["Text"].apply(lambda text: len(str(text).split()))
    df = df[(df["word_count"] >= min_words) & (df["word_count"] <= max_words)]

    stop_words = load_stopwords()
    df["sentiment"] = df["Score"].apply(get_sentiment)
    df["clean_text"] = df["Text"].apply(lambda text: clean_text(text, stop_words))
    return df


def prepare_train_test_data(
    csv_path,
    sample_size=10000,
    vocab_size=10000,
    max_length=100,
    test_size=0.2,
    random_state=42,
):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer

    df = load_and_clean_reviews(
        csv_path=csv_path,
        sample_size=sample_size,
        random_state=random_state,
    )

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["clean_text"])

    sequences = tokenizer.texts_to_sequences(df["clean_text"])
    x = pad_sequences(
        sequences,
        maxlen=max_length,
        padding="post",
        truncating="post",
    )
    y = df["sentiment"].map(LABEL_MAP).values

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return x_train, x_test, y_train, y_test, tokenizer, df


def parse_args():
    parser = argparse.ArgumentParser(description="Preview preprocessing for the review dataset.")
    parser.add_argument("--data", default="Reviews.csv", help="Path to Reviews.csv")
    parser.add_argument("--sample-size", type=int, default=10000)
    return parser.parse_args()


def main():
    args = parse_args()
    df = load_and_clean_reviews(csv_path=args.data, sample_size=args.sample_size)

    print("Cleaned dataset shape:", df.shape)
    print()
    print("Sentiment distribution:")
    print(df["sentiment"].value_counts())
    print()
    print("Preview:")
    print(df[["Text", "Score", "sentiment", "clean_text"]].head())


if __name__ == "__main__":
    main()
