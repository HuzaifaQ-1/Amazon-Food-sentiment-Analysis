# Results

## Experiment Setup

- Dataset: Amazon Fine Food Reviews
- Initial sample size: 10,000 reviews
- Final filtered dataset size: 5,556 reviews
- Task: 3-class sentiment classification
- Labels: negative, neutral, positive
- Train/test split: 80/20
- Training examples: 4,444
- Test examples: 1,112
- Vocabulary size: 10,000
- Maximum sequence length: 100
- Epochs: 5
- Batch size: 32

## Label Distribution

After filtering reviews by word count, the dataset was imbalanced toward positive reviews.

| Sentiment | Count | Share |
|---|---:|---:|
| Positive | 4,191 | 75.43% |
| Negative | 879 | 15.82% |
| Neutral | 486 | 8.75% |

## Models

| Model | Description |
|---|---|
| LSTM | Embedding layer, LSTM layer, ReLU activation, dropout, global max pooling, batch normalization, dense layer, softmax output |
| Simple RNN | Embedding layer, SimpleRNN layer, ReLU activation, dropout, batch normalization, dense layer, softmax output |

## Overall Metrics

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 Score |
|---|---:|---:|---:|---:|
| LSTM | 0.7365 | 0.7800 | 0.7400 | 0.7400 |
| Simple RNN | 0.7743 | 0.5995 | 0.7743 | 0.6758 |

## LSTM Class-Level Results

| Class | Precision | Recall | F1 Score | Support |
|---|---:|---:|---:|---:|
| Negative | 0.39 | 0.82 | 0.53 | 163 |
| Neutral | 0.29 | 0.06 | 0.10 | 88 |
| Positive | 0.90 | 0.79 | 0.84 | 861 |

The LSTM performed best on positive reviews, which is expected because the dataset contains many more positive examples. It also identified many negative reviews, but neutral sentiment remained difficult.

## Simple RNN Notes

The Simple RNN reached 0.7743 accuracy, but this score should be interpreted carefully. The notebook produced an undefined precision warning because at least one class had no predicted samples. Since the test set is heavily positive, the RNN can achieve high accuracy by favoring the majority class.

This means the Simple RNN accuracy is not enough by itself. Its weighted F1 score of 0.6758 gives a more realistic view of performance.

## Interpretation

The LSTM is the stronger model for this project despite having lower accuracy than the Simple RNN. Its class-level report shows more meaningful behavior across sentiment classes, especially for negative and positive reviews.

The Simple RNN appears to learn a simpler decision pattern and is more affected by class imbalance. This is visible from its high recall/accuracy but lower weighted precision and F1 score.

## Key Takeaways

- Accuracy alone is misleading on this dataset because positive reviews dominate the labels.
- Weighted F1 score and class-level precision/recall are more useful for model comparison.
- Neutral reviews are the hardest class for the current models.
- The LSTM is better suited than the Simple RNN for this sentiment classification task.

## Limitations

- The dataset is imbalanced toward positive reviews.
- The experiment uses a sampled subset for faster training.
- Stopword removal may remove words that matter for sentiment context.
- The models use learned embeddings from scratch instead of pretrained embeddings.
- Hyperparameters have not been extensively tuned.

## Future Improvements

- Add class weights or resampling to reduce majority-class bias.
- Add a Bidirectional LSTM model.
- Compare against classical baselines such as Logistic Regression and Naive Bayes.
- Use pretrained embeddings such as GloVe.
- Add confusion matrices and training curves to `reports/figures/`.
