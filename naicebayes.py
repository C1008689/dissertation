import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load the dataset
file_path = '/Users/aqib/Desktop/YEAR3/project/2500.json' 
data = []
with open(file_path, 'r') as file:
    for line in file:
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"Error decoding JSON for line: {line}")

# Create a DataFrame
df = pd.DataFrame(data)

# Extract features and labels
texts = df['reviewText']
labels = df['overall']

# Split the data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Vectorization and Model Training
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model on the training data
model.fit(train_texts, train_labels)

# Predict on the test data
predicted_labels = model.predict(test_texts)

# Classification report
report = classification_report(test_labels, predicted_labels)
print(report)
