import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# 1. Load the cleaned data
data = pd.read_csv("cleaned_emails.csv")
data = data.dropna(subset=["clean_text"])

X = data["clean_text"]
y = data["label"]

# 2. Split: 80% train, 20% test (stratify keeps the class balance equal)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training emails:", len(X_train))
print("Testing emails: ", len(X_test))

# 3. TF-IDF: turn text into numbers (fit on training data only)
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print("TF-IDF matrix shape:", X_train_tfidf.shape)

# 4. Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# 5. Evaluate on the unseen test set
y_pred = model.predict(X_test_tfidf)

print("\nRESULTS")
print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall   :", round(recall_score(y_test, y_pred), 4))
print("F1-score :", round(f1_score(y_test, y_pred), 4))

print("\nCONFUSION MATRIX (rows = actual, columns = predicted):")
print(confusion_matrix(y_test, y_pred))

print("\nFULL REPORT:")
print(classification_report(y_test, y_pred, target_names=["Safe", "Phishing"]))
