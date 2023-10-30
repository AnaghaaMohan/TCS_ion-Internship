# Import necessary libraries
import streamlit as st
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM model
model_path = 'lstm_model.pkl'  
with open(model_path, 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Load other resources needed for preprocessing, e.g., tokenizer
tokenizer_path = 'tokenizer.pkl'  
with open(tokenizer_path, 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Define the maximum sequence length for your model
MAX_SEQUENCE_LENGTH = 200
# Define a function to preprocess text and make predictions
def predict_sentiment(text):
    # Tokenize and pad the input text
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    # Make predictions
    prediction = loaded_model.predict(padded_sequences)[0][0]
    sentiment = "Positive" if prediction >= 0.5 else "Negative"

    return sentiment, prediction

# Streamlit UI
st.set_page_config(page_title="Movie Review Sentiment Analysis")
st.title("Movie Review Sentiment Analysis")

user_input = st.text_area("Enter your movie review:")
if st.button("Analyze"):
    if user_input:
        sentiment, confidence = predict_sentiment(user_input)
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Confidence: {confidence:.2f}")


