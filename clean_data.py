import pandas as pd
import re

data = pd.read_csv("Phishing_Email.csv")

# 1. Drop the useless index column
data = data.drop(columns=["Unnamed: 0"])

# 2. Drop rows with missing text
data = data.dropna(subset=["Email Text"])

# 3. Drop exact duplicate emails
before = len(data)
data = data.drop_duplicates(subset=["Email Text"])
print("Duplicates removed:", before - len(data))

# 4. Convert labels to numbers: Phishing = 1, Safe = 0
data["label"] = data["Email Type"].map({"Safe Email": 0, "Phishing Email": 1})

# 5. Clean the text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)   # replace links
    text = re.sub(r"\S+@\S+", " emailaddr ", text)    # replace email addresses
    text = re.sub(r"[^a-z\s]", " ", text)             # keep letters only
    text = re.sub(r"\s+", " ", text).strip()          # tidy spaces
    return text

data["clean_text"] = data["Email Text"].apply(clean_text)

# 6. Drop emails that became empty after cleaning
data = data[data["clean_text"] != ""]

# 7. Save the cleaned dataset
data[["clean_text", "label"]].to_csv("cleaned_emails.csv", index=False)

print("\nFinal size:", data.shape)
print("\nLabel counts:")
print(data["label"].value_counts())
print("\nExample cleaned email:")
print(data["clean_text"].iloc[0][:300])
