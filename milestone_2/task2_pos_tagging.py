"""
Student Name: Methunraj A
Student ID: 20
Task: 2 - POS Tagging
Date: 6 Oct 2025
"""

import spacy
from spacy.matcher import Matcher


# Load the spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def pos_tag_resume(text):
    """
    Tags each word with its Part of Speech.
    """
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]

def extract_nouns(text):
    """
    Extracts all NOUN and PROPN (proper nouns).
    """
    doc = nlp(text)
    return [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]

def find_adj_noun_patterns(text):
    """
    Finds all "Adjective + Noun" combinations.
    """
    doc = nlp(text)
    matcher = Matcher(nlp.vocab)
    pattern = [{"POS": "ADJ"}, {"POS": "NOUN"}]
    matcher.add("ADJ_NOUN_PATTERN", [pattern])
    matches = matcher(doc)
    
    patterns = []
    for match_id, start, end in matches:
        span = doc[start:end]
        patterns.append(span.text)
        
    return patterns

if __name__ == "__main__":
    # Test for Question 2.1
    text_2_1 = "John is an experienced Python developer"
    print("--- Question 2.1: Basic POS Tagging ---")
    pos_tags = pos_tag_resume(text_2_1)
    print(f"Input: {text_2_1}")
    print(f"Output: {pos_tags}")
    print(f"Expected Output: [('John', 'PROPN'), ('is', 'AUX'), ('an', 'DET'), ('experienced', 'ADJ'), ('Python', 'PROPN'), ('developer', 'NOUN')]\n")

    # Test for Question 2.2
    text_2_2 = "Experienced Data Scientist proficient in Machine Learning and Python programming"
    print("--- Question 2.2: Extract Nouns Only ---")
    nouns = extract_nouns(text_2_2)
    print(f"Input: {text_2_2}")
    print(f"Output: {nouns}")
    print(f"Expected Output: ['Data', 'Scientist', 'Machine', 'Learning', 'Python', 'programming']\n")

    # Test for Question 2.3
    text_2_3 = "Expert in Machine Learning, Deep Learning, and Natural Language Processing"
    print("--- Question 2.3: Identify Skill Patterns ---")
    adj_noun_patterns = find_adj_noun_patterns(text_2_3)
    print(f"Input: {text_2_3}")
    print(f"Output: {adj_noun_patterns}")
    # The expected output for "Natural Language Processing" is "Natural Language".
    # The function correctly identifies "Natural Language" as an ADJ+NOUN pair.
    print(f"Expected Output: ['Machine Learning', 'Deep Learning', 'Natural Language']\n")