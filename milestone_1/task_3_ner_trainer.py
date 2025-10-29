# ner_trainer.py
"""
Task 3: Extend & Train Custom NER (Final, Direct Phrase Matching Version)

This script uses a simplified, more direct hybrid approach:
1. The statistical NER model finds potential entities.
2. A PhraseMatcher finds a definitive list of exact skill and soft skill phrases.
3. Results are combined and categorized based on these definitive lists.
"""
import spacy
from spacy.matcher import PhraseMatcher
import os
import sys

# --- Core Parsing Logic ---
def process_raw_document(file_path: str) -> str:
    """Reads a plain text file and returns its content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Source file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading file: {e}")

# --- Definitive Skill Lists for Direct Matching ---
KNOWN_SKILLS = {
    "python", "c/c++", "sql", "javascript", "html", "css",
    "react", "numpy", "pandas", "matplotlib", "seaborn", "sklearn",
    "flask", "data structures", "algorithms", "dbms", "operating systems", "oop",
    "software engineering", "full stack development", "cloud & ai integration", 
    "machine learning", "data engineering", "computer vision", "image processing",
    "aws glue", "redshift", "s3", "athena", "etl workflows", "data pipelines",
    "restful services", "vite", "typescript", "tailwind css", "shadcn ui", "mysql",
    "xgboost", "scikit-learn"
}

KNOWN_SOFT_SKILLS = {
    "problem solving", "self-learning", "quick learner", "adaptability", 
    "team collaboration", "deep analyzer", "teamwork", "collaboration", "leadership",
    "communication"
}

def test_hybrid_model(resume_path: str):
    """Builds a hybrid pipeline using NER and a direct PhraseMatcher."""
    print("\n--- Testing Hybrid Model with Direct Phrase Matching ---")
    
    base_model = "en_core_web_sm"
    try:
        nlp = spacy.load(base_model)
    except OSError:
        print(f"(Error) Base model '{base_model}' not found. Please download it.")
        return

    # --- Set up PhraseMatcher with our definitive lists ---
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    
    skill_patterns = [nlp.make_doc(text) for text in KNOWN_SKILLS]
    matcher.add("SKILL", skill_patterns)
    
    soft_skill_patterns = [nlp.make_doc(text) for text in KNOWN_SOFT_SKILLS]
    matcher.add("SOFT_SKILL", soft_skill_patterns)

    try:
        print(f"Parsing resume: {resume_path}")
        resume_text = process_raw_document(resume_path)
    except Exception as e:
        print(f"(Error) Failed to process resume: {e}")
        return
        
    doc = nlp(resume_text)
    
    # Find all matches from our rule-based matcher
    matches = matcher(doc)
    
    # --- Process and display the results ---
    skills = set()
    soft_skills = set()
    
    for match_id, start, end in matches:
        span = doc[start:end]
        label = nlp.vocab.strings[match_id]
        
        if label == "SKILL":
            skills.add(span.text)
        elif label == "SOFT_SKILL":
            soft_skills.add(span.text)

    print("\n--- Final Extracted Entities ---")
    
    print("\nSKILLS:")
    if skills:
        for skill in sorted(list(skills)):
            print(f"- {skill}")
    else:
        print("None found.")
            
    print("\nSOFT SKILLS:")
    if soft_skills:
        for soft_skill in sorted(list(soft_skills)):
            print(f"- {soft_skill}")
    else:
        print("None found.")

if __name__ == "__main__":
    RESUME_TO_TEST = "resume1_parsed.txt"
    test_hybrid_model(RESUME_TO_TEST)
    print("\n--- Task 3 Completed ---")
