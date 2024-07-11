import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Sample review text
review_text = "Water for Elephants has a quality so many modern novels are lacking, and that is an sense of enchantment..."

# Lowercase the text
review_text = review_text.lower()

# Remove punctuation
review_text = re.sub(r'[^\w\s]', '', review_text)

# Tokenize the text
tokens = word_tokenize(review_text)

# Remove stop words
stop_words = set(stopwords.words('english'))
tokens = [word for word in tokens if word not in stop_words]

# Lemmatize the words
lemmatizer = WordNetLemmatizer()
tokens = [lemmatizer.lemmatize(word) for word in tokens]

# Join tokens back to string
preprocessed_text = ' '.join(tokens)
print(preprocessed_text)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Sample data
texts = ["Water for Elephants has a quality...", "Another review text..."]
labels = [1, 0]  # Example labels (1 for positive, 0 for negative)

# Tokenizing the texts
tokenizer = Tokenizer(num_words=5000)  # Considering top 5000 words
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

# Padding sequences to ensure uniform input size
data = pad_sequences(sequences, maxlen=100)