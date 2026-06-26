import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

from evaluate import evaluate_predictions
from preprocessing import prepare_train_test_data


def build_lstm_model(vocab_size=10000, max_length=100):
    from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Dropout, Embedding, GlobalMaxPooling1D, LSTM
    from tensorflow.keras.models import Sequential

    model = Sequential(
        [
            Embedding(input_dim=vocab_size, output_dim=128),
            LSTM(64, return_sequences=True),
            Activation("relu"),
            Dropout(0.5),
            GlobalMaxPooling1D(),
            BatchNormalization(),
            Dense(64, activation="relu"),
            Dense(3, activation="softmax"),
        ]
    )

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train an LSTM sentiment classifier.")
    parser.add_argument("--data", default="Reviews.csv", help="Path to Reviews.csv")
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--max-length", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-model", default=None, help="Optional path to save the trained model")
    return parser.parse_args()


def main():
    args = parse_args()

    x_train, x_test, y_train, y_test, _, _ = prepare_train_test_data(
        csv_path=args.data,
        sample_size=args.sample_size,
        vocab_size=args.vocab_size,
        max_length=args.max_length,
    )

    model = build_lstm_model(
        vocab_size=args.vocab_size,
        max_length=args.max_length,
    )

    model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(x_test, y_test),
    )

    y_pred = model.predict(x_test).argmax(axis=1)
    evaluate_predictions(y_test, y_pred, model_name="LSTM")

    if args.save_model:
        model.save(args.save_model)


if __name__ == "__main__":
    main()
