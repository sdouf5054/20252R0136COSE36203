
from google.colab import files
uploaded = files.upload()

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import RandomOverSampler

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

import torch
import os

os.environ["WANDB_DISABLED"] = "true"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on device:", device)

df = pd.read_csv("youtube_videos_big.csv")
print("Dataset Loaded:", df.shape)

mapping = {
    10: 'Music', 17: 'Sports', 19: 'Travel', 20: 'Gaming',
    22: 'People & Blogs', 23: 'Comedy', 24: 'Entertainment',
    25: 'News & Politics', 26: 'HowTo & Style',
    27: 'Education', 28: 'Science & Tech'
}

df['category_group'] = df['category_id'].map(mapping)
df = df.dropna(subset=['category_group'])

df['text'] = (
    df['title'].fillna('') + " " +
    df['description'].fillna('') + " " +
    df['keyword'].fillna('')
)

df = df[['text', 'category_group']]
print("Final usable dataset:", df.shape)

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['category_group'],
    test_size=0.2,
    stratify=df['category_group'],
    random_state=42
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))

tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train_vec, y_train)

print("Oversampled train size:", X_train_resampled.shape[0])

model_lr = LogisticRegression(max_iter=4000, class_weight="balanced")
model_lr.fit(X_train_resampled, y_train_resampled)
pred_lr = model_lr.predict(X_test_vec)

print("\n========= TF-IDF BASELINE RESULTS =======%%")
print("Baseline Accuracy:", accuracy_score(y_test, pred_lr))
print(classification_report(y_test, pred_lr))

cm = confusion_matrix(y_test, pred_lr)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, cmap="Blues")
plt.title("TF-IDF Baseline Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

le = LabelEncoder()
df['label'] = le.fit_transform(df['category_group'])

train_df, test_df = train_test_split(
    df[['text','label']],
    test_size=0.2,
    stratify=df['label'],
    random_state=42
)

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(
    train_df[['text']],
    train_df['label']
)

train_df_resampled = pd.DataFrame({
    'text': X_resampled['text'],
    'label': y_resampled
})

num_labels = df['label'].nunique()

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

from datasets import Dataset

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

ds_train = Dataset.from_pandas(train_df_resampled).map(tokenize, batched=True)
ds_test = Dataset.from_pandas(test_df).map(tokenize, batched=True)

ds_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
ds_test.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

model_bert = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=num_labels
).to(device)

training_args = TrainingArguments(
    output_dir="./bert_results",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    logging_dir="./logs"
)

trainer = Trainer(
    model=model_bert,
    args=training_args,
    train_dataset=ds_train,
    eval_dataset=ds_test
)

trainer.train()

preds = trainer.predict(ds_test)
y_pred = np.argmax(preds.predictions, axis=1)

print("\n========= IMPROVED BERT RESULTS =========")
print("Improved BERT Accuracy:", accuracy_score(test_df['label'], y_pred))
print(classification_report(test_df['label'], y_pred, target_names=le.classes_))

examples = [
    "Robot chef cooking pasta using AI",
    "Breaking news: government announces new education policy",
    "Relaxing jazz music playlist",
    "Beginner Python tutorial",
    "Gaming laptop review"
]

print("\n========= EXAMPLE PREDICTIONS =========")
for text in examples:
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
    output = model_bert(**enc)
    pred = torch.argmax(output.logits, dim=1).item()
    print(text, "→", le.inverse_transform([pred])[0])
