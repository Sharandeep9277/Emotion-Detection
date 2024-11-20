# Importing libraries
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from flask import Flask, request, jsonify
import tensorflow as tf

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

from flask_cors import CORS


# Load dataset
df = pd.read_csv('Emotion_classify_Data.csv')  # Replace with your dataset path
df.columns = ['text', 'emotion']  # Ensure columns are named correctly
print(df.head())

# Data distribution visualization
sns.countplot(x='emotion', data=df)
plt.title('Emotion Distribution')
plt.show()

# Text cleaning function
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)    # Remove mentions
    text = re.sub(r"#", "", text)       # Remove hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text

df['cleaned_text'] = df['text'].apply(clean_text)

# Tokenization and lemmatization
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

df['processed_text'] = df['cleaned_text'].apply(preprocess)
print(df.head())

# Splitting data into train and test sets
X = df['processed_text']
y = df['emotion']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Label encoding
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train).toarray()
X_test_tfidf = tfidf.transform(X_test).toarray()

# Dimensionality reduction with PCA
pca = PCA(n_components=300)
X_train_pca = pca.fit_transform(X_train_tfidf)
X_test_pca = pca.transform(X_test_tfidf)

# Model creation
model = Sequential([
    Dense(128, activation='relu', input_dim=X_train_pca.shape[1]),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(encoder.classes_), activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_pca, y_train, epochs=20, batch_size=32, validation_data=(X_test_pca, y_test))

# Plot accuracy and loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.show()

# Classification report and confusion matrix
y_pred = np.argmax(model.predict(X_test_pca), axis=1)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title('Confusion Matrix')
plt.show()

# Save the model and preprocessing objects
model.save('emotion_detection_modell.h5')
joblib.dump(encoder, 'label_encoder.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(pca, 'pca_model.pkl')

# Deployment using Flask (basic setup)
app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

# Load the saved model and preprocessing objects
loaded_model = tf.keras.models.load_model('emotion_detection_modell.h5')
encoder = joblib.load('label_encoder.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
pca = joblib.load('pca_model.pkl')

# Preprocess the input text
def preprocess_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)    # Remove mentions
    text = re.sub(r"#", "", text)       # Remove hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get('text')
    if not data:
        return jsonify({'error': 'No text provided'}), 400

    cleaned = preprocess_text(data)
    vectorized = tfidf.transform([cleaned]).toarray()
    reduced = pca.transform(vectorized)
    prediction = np.argmax(loaded_model.predict(reduced), axis=1)
    emotion = encoder.inverse_transform(prediction)[0]
    return jsonify({'emotion': emotion})

# Home route to check if the server is running
@app.route('/')
def home():
    return "Emotion Detection API is running"

if __name__ == "__main__":
    app.run(debug=True)
