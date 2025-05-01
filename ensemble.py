from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import joblib



import pandas as pd

#Load data set
df = pd.read_csv("emails.csv")

# Split data set columns
#email_contents = df['text']
#labels = df['spam']

# Split your data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['spam'], test_size=0.2, random_state=42
)

# Create base classifiers
clf1 = LogisticRegression(max_iter=1000)
clf2 = MultinomialNB()
clf3 = SVC(probability=True)  # SVC needs probability=True for soft voting

# Ensemble: Voting Classifier
voting_clf = VotingClassifier(
    estimators=[
        ('lr', clf1),
        ('nb', clf2),
        ('svc', clf3)
    ],
    voting='soft'  # Use 'hard' for majority vote if no probabilities
)

# Full pipeline
ensemble_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('voting_clf', voting_clf)
])

# Train and evaluate
ensemble_pipeline.fit(X_train, y_train)
y_pred = ensemble_pipeline.predict(X_test)

print(classification_report(y_test, y_pred))


#Save the model for using it in the app
joblib.dump(ensemble_pipeline, "spam_ensemble.pkl")

# Plot the results

metrics = {
    'Precision': [0.99, 0.99],
    'Recall':    [1.00, 0.96],
    'F1-Score':  [0.99, 0.98]
}

classes = ['Not Spam (0)', 'Spam (1)']
x = np.arange(len(classes))
width = 0.2

fig, ax = plt.subplots(figsize=(8, 5))

for i, (metric_name, scores) in enumerate(metrics.items()):
    ax.bar(x + i*width, scores, width, label=metric_name)

ax.set_ylabel('Score')
ax.set_title('Ensemble Model Performance by Class')
ax.set_xticks(x + width)
ax.set_xticklabels(classes)
ax.set_ylim(0.90, 1.05)
ax.legend()
plt.tight_layout()
plt.show()


# Plot the confusion matrix

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Spam', 'Spam'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - Ensemble Model")
plt.show()
