import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
import joblib

# Download necessary NLTK datasets
nltk.download('punkt')
nltk.download('stopwords')

# Set of English stop words
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenize and filter out stop words
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    return " ".join(filtered_sentence)

def process_line(line):
    # Load JSON data and extract the review text and overall score
    json_data = json.loads(line)
    text = json_data.get('reviewText', '')
    overall = json_data.get('overall', 0)
    # Define sentiment: 1 for positive (rating > 3) and 0 for negative
    sentiment = 1 if overall > 3 else 0
    return preprocess_text(text), sentiment

# Initialize the vectorizer and the SGDClassifier
vectorizer = HashingVectorizer(stop_words=stop_words, n_features=2**18, alternate_sign=False)
classifier = SGDClassifier(loss='log_loss')  # If the documentation indicates this is correct


# Path to your JSON file
file_path = '/Users/aqibullah/Downloads/Books_5.json'

# Process the file in chunks to manage memory usage
chunk_size = 1000
X_chunk, y_chunk = [], []

with open(file_path, 'r', encoding='utf-8') as file:
    for i, line in enumerate(file):
        # Process each line in the file
        X_line, y_line = process_line(line)
        X_chunk.append(X_line)
        y_chunk.append(y_line)
        
        # When the chunk is full, vectorize and perform partial fitting
        if len(X_chunk) == chunk_size:
            X_vectorized = vectorizer.transform(X_chunk)
            classifier.partial_fit(X_vectorized, y_chunk, classes=[0, 1])
            X_chunk, y_chunk = [], []  # Clear the chunk

        # Print progress every 10,000 lines
        if i % 10000 == 0:
            print(f"Processed {i} lines.")

# Make sure the last chunk is processed
if X_chunk:
    X_vectorized = vectorizer.transform(X_chunk)
    classifier.partial_fit(X_vectorized, y_chunk, classes=[0, 1])

# Save the classifier to disk
joblib.dump(classifier, 'sentiment_classifier.pkl')

print("Training complete.")
