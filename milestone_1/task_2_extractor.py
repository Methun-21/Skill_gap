# task_2_extractor.py
"""
Task 2: Skill Dictionary & Rule-Based Extraction

Instructions:
1. Defines a dictionary of 50 skills in a separate file: skills_dict.txt.
2. Writes a function extract_skills(text) that uses PhraseMatcher to find skills.
3. Runs it on one parsed resume and the JD.
4. Deliverable: Print two lists -> Resume skills, JD skills.
"""
import spacy
from spacy.matcher import PhraseMatcher
from typing import List, Dict
import os
import sys
import json
from io import BytesIO

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Import Core Processing Logic (Assuming complete_pipeline.py is in the same directory) ---
try:
    from complete_pipeline import TextExtractor, TextCleaner
except ImportError:
    print("Error: Could not import TextExtractor/TextCleaner from complete_pipeline.py.")
    print("Please ensure complete_pipeline.py is in the parent directory.")
    sys.exit(1)

# --- Core Parsing Function ---

def process_raw_document(file_content: bytes, file_name: str, doc_type: str) -> str:
    """Runs the full extraction and cleaning pipeline from complete_pipeline.py."""
    extractor = TextExtractor()
    cleaner = TextCleaner()

    file_format = file_name.split('.')[-1].lower()
    file_info = {'name': file_name, 'content': file_content, 'format': file_format}

    extraction = extractor.extract_text(file_info)
    if not extraction['success']:
        raise Exception(f"Extraction failed for {file_name}: {extraction['error']}")

    cleaning = cleaner.clean_text(extraction['text'], doc_type)
    if not cleaning['success']:
        raise Exception(f"Cleaning failed for {file_name}: {cleaning['error']}")

    return cleaning['cleaned_text']


# --- Skill Extraction Logic ---

def extract_skills(text: str, skill_list: List[str]) -> List[str]:
    """
    Initializes a spaCy PhraseMatcher and extracts skills from the provided text.
    """
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model 'en_core_web_sm'...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    matcher = PhraseMatcher(nlp.vocab)

    patterns = [nlp.make_doc(skill) for skill in skill_list]
    matcher.add("SkillPattern", patterns)

    doc = nlp(text)
    matches = matcher(doc)

    found_skills = set()
    for match_id, start, end in matches:
        span = doc[start:end]
        found_skills.add(span.text)

    return sorted(list(found_skills))

def load_parsed_text(filepath: str) -> str:
    """Helper function to load cleaned text from file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}.")
        print("Please ensure you have run Task 1 to create these files.")
        return ""

def load_skills_from_file(filepath: str) -> List[str]:
    """Loads a list of skills from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            skill_data = json.load(f)
            all_skills = []
            for category in skill_data:
                all_skills.extend(skill_data[category])
            return all_skills
    except FileNotFoundError:
        print(f"Error: Skill dictionary file not found at {filepath}.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}.")
        return []


def run_task_2():
    # --- File paths from Task 1 ---
    RESUME_FILE = 'resume1_parsed.txt'
    JD_FILE = 'jd_parsed.txt'
    SKILL_DICT_FILE = '../milestone_2/skill_database.txt'

    # Load skills from the file
    skill_dictionary = load_skills_from_file(SKILL_DICT_FILE)
    if not skill_dictionary:
        return

    # Load content from parsed files
    resume_text = load_parsed_text(RESUME_FILE)
    jd_text = load_parsed_text(JD_FILE)

    if not resume_text or not jd_text:
        return

    print("\n--- Starting Task 2: Rule-Based Skill Extraction ---")

    # Run matcher on resume
    resume_skills = extract_skills(resume_text, skill_dictionary)

    # Run matcher on JD
    jd_skills = extract_skills(jd_text, skill_dictionary)

    # Deliverable: Print two lists
    print(f"\n--- Resume Skills ({RESUME_FILE}) ---")
    print(f"Total Found: {len(resume_skills)}")
    print(resume_skills)

    print(f"\n--- JD Skills ({JD_FILE}) ---")
    print(f"Total Found: {len(jd_skills)}")
    print(jd_skills)

    print("\n--- Task 2 Completed ---")

if __name__ == '__main__':
    run_task_2()