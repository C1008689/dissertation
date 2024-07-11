import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import numpy as np

# File path to the dataset
file_path = '/Users/aqib/Desktop/YEAR3/project/2500.json'

# Load the JSON data
with open(file_path, 'r') as file:
    data = json.load(file)
    
review_text = data 

# Extract texts and labels
texts = [review['text'] for review in data]
labels = [review['label'] for review in data]

# Preprocess the texts
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

preprocessed_texts = [preprocess_text(text) for text in texts]

# Tokenize the texts
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(preprocessed_texts)
sequences = tokenizer.texts_to_sequences(preprocessed_texts)
word_index = tokenizer.word_index

# Pad sequences
data = pad_sequences(sequences, maxlen=100)

# Build the model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=100))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Convert data and labels to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Train the model
model.fit(data, labels, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(data, labels)
print(f'Loss: {loss}, Accuracy: {accuracy}')