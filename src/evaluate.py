from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score


def evaluate_predictions(y_true, y_pred, model_name="Model"):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    print(f"{model_name} Accuracy: {metrics['accuracy']:.4f}")
    print(f"{model_name} Precision: {metrics['precision']:.4f}")
    print(f"{model_name} Recall: {metrics['recall']:.4f}")
    print(f"{model_name} F1 Score: {metrics['f1']:.4f}")
    print()
    print(classification_report(y_true, y_pred, zero_division=0))

    return metrics


if __name__ == "__main__":
    print(
        "evaluate.py contains helper functions for model evaluation.\n"
        "Run a training script instead, for example:\n"
        "  python src/train_lstm.py --data Reviews.csv\n"
        "  python src/train_rnn.py --data Reviews.csv"
    )
