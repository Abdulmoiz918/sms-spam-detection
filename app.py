import numpy as np
import streamlit as st
import pickle
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem.porter import PorterStemmer

# Initialize the Porter Stemmer
ps = PorterStemmer()

# Define the text transformation function
def transform_text(text):
    text = text.lower()
    
    # Basic tokenization using split
    text = text.split()
    
    # Remove non-alphanumeric characters and stopwords
    stop_words = set(ENGLISH_STOP_WORDS)
    y = [word for word in text if word.isalnum()]
    text = [word for word in y if word not in stop_words and word not in string.punctuation]
    
    # Apply stemming
    text = [ps.stem(word) for word in text]
    
    return " ".join(text)

# Load your trained vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit code to get user input
st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. Preprocess the input text
    transformed_sms = transform_text(input_sms)
    
    # 2. Vectorize the preprocessed text
    vector_input = tfidf.transform([transformed_sms]).toarray()
    
    # 3. Append the 'num_characters' feature to the vector_input
    num_characters = len(input_sms)
    vector_input = np.hstack((vector_input, [[num_characters]]))

    # 4. Predict using the model
    result = model.predict(vector_input)[0]
    
    # 5. Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
