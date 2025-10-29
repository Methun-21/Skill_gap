"""
Student Name: Methunraj A
Student ID: 20
Task: Bonus Tasks
Date: 6 Oct 2025
"""

import spacy
import json
import re
from collections import Counter

nlp = spacy.load("en_core_web_sm")

def count_skill_frequency(text: str, skills_to_find: list) -> str:
    skill_counts = Counter()
    for skill in skills_to_find:
        found = re.findall(r"\b" + re.escape(skill) + r"\b", text, re.IGNORECASE)
        if found:
            skill_counts[skill] += len(found)
    sorted_skills = skill_counts.most_common()
    output = []
    for skill, count in sorted_skills:
        output.append(f"{skill}: {count} times")
    return "\n".join(output)

# Extracts the context in which a skill is mentioned.
def extract_skill_context(text: str, skill: str) -> str:
    doc = nlp(text)
    context_sentences = []
    for sent in doc.sents:
        if skill.lower() in sent.text.lower():
            context_sentences.append(sent.text.strip())
    output = [f"Skill: {skill}"]
    for i, sentence in enumerate(context_sentences, 1):
        output.append(f'Context {i}: "{sentence}"')
    return "\n".join(output)

if __name__ == "__main__":
    print("--- Testing Bonus 1: Skill Frequency Counter ---")
    bonus_1_text = """
    Python developer with Python experience. 
    Used Python and Machine Learning. 
    Machine Learning projects with Python.
    """
    skills_to_test = ["Python", "Machine Learning", "Java"]
    frequency_output = count_skill_frequency(bonus_1_text, skills_to_test)
    print(frequency_output)
    print("--- Testing Bonus 2: Skill Context Extractor ---")
    bonus_2_text = """
    I am a Python developer. I have 5 years of experience in Python.
    Also worked on Java projects. Python is my primary language.
    """
    skill_to_find = "Python"
    context_output = extract_skill_context(bonus_2_text, skill_to_find)
    print(context_output)