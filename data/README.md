# Dataset

## Source

This project uses the Amazon Fine Food Reviews dataset, commonly distributed as `Reviews.csv`.

Suggested source:

- Kaggle: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

## Required Columns

The original file contains multiple metadata columns:

```text
Id, ProductId, UserId, ProfileName, HelpfulnessNumerator,
HelpfulnessDenominator, Score, Time, Summary, Text
```

This project only uses:

| Column | Description |
|---|---|
| `Text` | Full review text used as the model input |
| `Score` | Review rating from 1 to 5, used to create sentiment labels |

## Label Mapping

Sentiment labels are created from the review score:

| Score | Sentiment | Encoded Label |
|---:|---|---:|
| 1-2 | Negative | 0 |
| 3 | Neutral | 1 |
| 4-5 | Positive | 2 |

## Filtering and Sampling

The experiment samples 10,000 reviews for faster undergraduate-level experimentation. After sampling, reviews are filtered by word count:

| Filter | Value |
|---|---:|
| Minimum words | 50 |
| Maximum words | 500 |

In the current notebook run, this produced 5,556 usable reviews after filtering.
