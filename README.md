# Email Phishing Detection

A machine learning project that classifies emails as **Phishing** or **Safe** using TF-IDF features and Logistic Regression.

## Dataset

Dataset: [Phishing Email Dataset on Kaggle](https://www.kaggle.com/datasets/subhajournal/phishingemails) (`Phishing_Email.csv`, 18,650 emails with the columns `Email Text` and `Email Type`).
The CSV is not included in this repository. Download it from Kaggle and place it in the project folder.

## Pipeline

1. **Clean** (`clean_data.py`): drop the index column, empty rows and duplicate emails; map labels to 0 (safe) and 1 (phishing); lowercase the text, replace links and email addresses with placeholders, and keep letters only. Output: `cleaned_emails.csv` (17,535 emails).
2. **Train and evaluate** (`train_model.py`): 80/20 stratified split, TF-IDF (5,000 features, English stop words removed), Logistic Regression.

## Results (test set, 3,507 emails)

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Safe | 0.97 | 0.99 | 0.98 |
| Phishing | 0.97 | 0.95 | 0.96 |
| **Accuracy** | | | **0.97** |

## How to run

```
pip install -r requirements.txt
python clean_data.py
python train_model.py
```

## Limitations

- The dataset mixes phishing with spam, and the safe emails include older mailing-list text, so the model may partly learn the style of the source.
- Results may not generalise to modern emails from other sources.

## Next steps

- Compare Naive Bayes, Linear SVM and Random Forest
- Try word pairs (n-grams) and more features
- Analyse the missed phishing emails
- Save the model with `joblib` and classify new emails
