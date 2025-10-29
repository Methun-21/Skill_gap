"""
Student Name: Methunraj A
Student ID: 20
Task: 1 - Preprocessing
Date: 6 Oct 2025
"""

import re
import spacy

# Load the spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def clean_resume_text(text):
    """
    Removes all email addresses, phone numbers, URLs, and special characters
    except (+ # - .), and converts text to lowercase.
    """
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove phone numbers
    text = re.sub(r'\+?\d[\d -]{8,12}\d', '', text)
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove special characters except (+ # - .)
    text = re.sub(r'[^a-zA-Z0-9\s\+\#\-\.]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def tokenize_text(text):
    """
    Splits text into individual words (tokens) using spaCy.
    """
    doc = nlp(text)
    tokens = [token.text for token in doc]
    return tokens

def remove_stop_words(text):
    """
    Removes common stop words, preserving specified programming language names.
    """
    preserved_words = {"c", "r", "go", "d"}
    doc = nlp(text)
    stop_words = spacy.lang.en.stop_words.STOP_WORDS
    filtered_tokens = [token.text for token in doc if token.text.lower() not in stop_words or token.text.lower() in preserved_words]
    return " ".join(filtered_tokens)

# Lemmatizes the text to its base form.
def lemmatize_text(text):
    """
    Converts words to their base form (lemma).
    """
    doc = nlp(text)
    lemmatized_tokens = [token.lemma_ for token in doc]
    return " ".join(lemmatized_tokens)

if __name__ == "__main__":
    text_1_1 = """
    Contact: john@email.com | Phone: +1-555-0123
    Visit: www.johndoe.com
    Skills: Python, C++, C#, .NET
    """
    print("--- Question 1.1: Basic Text Cleaning ---")
    cleaned_text = clean_resume_text(text_1_1)
    print(f"Input:\n{text_1_1}")
    print(f"Output: {cleaned_text}")
    text_1_2 = "I'm a Python developer. I've worked on ML projects."
    print("--- Question 1.2: Tokenization ---")
    tokens = tokenize_text(text_1_2)
    print(f"Input: {text_1_2}")
    print(f"Output: {tokens}")
    text_1_3 = "I have experience in Python and R programming with excellent skills in C and Go"
    print("--- Question 1.3: Stop Words Removal ---")
    no_stop_words_text = remove_stop_words(text_1_3)
    print(f"Input: {text_1_3}")
    print(f"Output: {no_stop_words_text}")
    text_1_4 = "I am working on developing multiple applications using programming languages"
    print("--- Question 1.4: Lemmatization ---")
    lemmatized_text = lemmatize_text(text_1_4)
    print(f"Input: {text_1_4}")
    print(f"Output: {lemmatized_text}")
