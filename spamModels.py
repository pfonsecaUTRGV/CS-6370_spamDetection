from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import joblib

import pandas as pd

#Load data set
df = pd.read_csv("emails.csv")

# Split data set columns
email_contents = df['text']
labels = df['spam']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    email_contents, labels, test_size=0.2, random_state=42)

# Classifier models to train and compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True)
}

# Build and evaluate pipelines
for name, model in models.items():
    print(f"\n{name}")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

# Save best model

joblib.dump(pipeline, 'spam_detector.pkl') 

# Plot results
results = {
    "Logistic Regression": {"accuracy": 0.98, "precision": 0.99, "recall": 0.91, "f1": 0.95},
    "Naive Bayes": {"accuracy": 0.89, "precision": 1.00, "recall": 0.58, "f1": 0.73},
    "Random Forest": {"accuracy": 0.98, "precision": 1.00, "recall": 0.92, "f1": 0.96},
    "SVM": {"accuracy": 0.99, "precision": 0.99, "recall": 0.97, "f1": 0.98}
}

models = list(results.keys())
metrics = ['accuracy', 'precision', 'recall', 'f1']

# Create a 4x1 list of lists for plotting
values = [[results[model][metric] for model in models] for metric in metrics]

# Plot
x = np.arange(len(models))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 6))
for i, (metric_vals, label) in enumerate(zip(values, metrics)):
    ax.bar(x + i*width, metric_vals, width, label=label)

ax.set_ylabel('Score')
ax.set_title('Classifier Performance on Spam Detection')
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(models, rotation=15)
ax.set_ylim(0.5, 1.05)
ax.legend()
plt.tight_layout()
plt.show()
