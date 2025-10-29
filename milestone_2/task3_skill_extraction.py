"""
Student Name: Methunraj A
Student ID: 20
Task: 3 - Skill Extraction
Date: 6 Oct 2025
"""

import json

def load_from_file(filename):
    with open(filename, 'r') as f:
        return json.load(f)

SKILL_DATABASE = load_from_file("skill_database.txt")
ABBREVIATIONS = load_from_file("abbreviations.txt")

def extract_skills(text, skill_database):
    found_skills = {category: [] for category in skill_database.keys()}
    for category, skills in skill_database.items():
        for skill in skills:
            if skill.lower() in text.lower():
                found_skills[category].append(skill)
    return {category: skills for category, skills in found_skills.items() if skills}

# Normalizes skill names using a predefined list of abbreviations.
def normalize_skills(skill_list):
    abbr_map = {k.lower(): v for k, v in ABBREVIATIONS.items()}
    normalized = []
    for skill in skill_list:
        if skill.lower() in abbr_map:
            normalized.append(abbr_map[skill.lower()])
        else:
            normalized.append(skill)
    return normalized

if __name__ == "__main__":
    text_input = "Proficient in Python, Java, TensorFlow, and AWS. Strong leadership skills."
    extracted = extract_skills(text_input, SKILL_DATABASE)
    print("Extracted Skills:")
    print(extracted)
    skills_input = ['ML', 'DL', 'NLP', 'JS', 'K8s', 'AWS', 'GCP']
    normalized = normalize_skills(skills_input)
    print("Normalized Skills:")
    print(normalized)
