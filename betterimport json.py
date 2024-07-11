import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load the dataset from JSON file
file_path = '/Users/aqib/Desktop/YEAR3/project/2500.json'
with open(file_path, 'r') as f:
    lines = f.readlines()

# Each line in the file is a separate JSON object, handle potential empty lines
data = []
for line in lines:
    line = line.strip()
    if line:
        data.append(json.loads(line))

# Convert to DataFrame
df = pd.DataFrame(data)

# Extract review texts and labels
texts = df['reviewText'].values
labels = df['overall'].values

# Preprocess labels
labels = labels - 1  # Adjust labels to be 0-4 for a 5-class classification

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
maxlen = 100
data = pad_sequences(sequences, maxlen=maxlen)

# Build the model
model = Sequential()
model.add(Embedding(10000, 128, input_length=maxlen))
model.add(SpatialDropout1D(0.4))  # Increased dropout
model.add(LSTM(128, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
model.add(LSTM(64, dropout=0.3, recurrent_dropout=0.3))
model.add(Dense(5, activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Callbacks for early stopping and saving the best model
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, min_delta=0.0001),
    ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')
]

# Train the model
history = model.fit(data, labels, epochs=20, batch_size=32, validation_split=0.2, callbacks=callbacks)

# Evaluate the model
accr = model.evaluate(data, labels)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))
