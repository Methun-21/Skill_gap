"""
Student Name: Methunraj A
Student ID: 20
Task: 5 - Complete Extractor
Date: 6 Oct 2025
"""

import spacy
import json
import re
from spacy.matcher import Matcher

# Load the small English language model for spaCy
nlp = spacy.load("en_core_web_sm")

# Load the skill database from a JSON file.
with open("skill_database.txt", 'r') as f:
    SKILL_DATABASE = json.load(f)

def extract_all_skills(resume_text):
    """
    Extracts technical and soft skills from a resume text using a multi-faceted approach.
    
    This function combines three methods:
    1. Database-based matching: Finds skills listed in a predefined dictionary.
    2. Pattern-based matching: Identifies multi-word skills using grammatical patterns (e.g., ADJ + NOUN).
    3. Named Entity Recognition (NER): Uses spaCy's pre-trained model to find products and organizations.
    4. Keyword-based matching: Hard-coded check for common phrases.

    Args:
        resume_text (str): The raw text of the resume.

    Returns:
        dict: A dictionary containing lists of technical skills, soft skills, and all skills.
    """
    technical_skills = set()
    soft_skills = set()

    # --- Method 1: Database Matching ---
    # Iterate through the skill database and use regular expressions to find
    # skills in the resume text.
    for category, skills in SKILL_DATABASE.items():
        for skill in skills:
            # Use re.search with word boundaries (\b) and case-insensitivity to find exact skill names.
            if re.search(r"\b" + re.escape(skill) + r"\b", resume_text, re.IGNORECASE):
                if category == 'soft_skills':
                    soft_skills.add(skill)
                else:
                    technical_skills.add(skill)

    # --- Method 2: Pattern Matching for Soft Skills ---
    # Process the text with spaCy to get a Doc object
    doc = nlp(resume_text)
    
    # Initialize the Matcher with the shared vocabulary
    matcher = Matcher(nlp.vocab)
    
    # Define patterns for two-word phrases (e.g., "strong leadership", "problem solving")
    patterns = [
        [{'POS': 'ADJ'}, {'POS': 'NOUN'}],  # Adjective followed by a Noun
        [{'POS': 'NOUN'}, {'POS': 'NOUN'}]   # Noun followed by a Noun
    ]
    matcher.add("SKILL_PATTERN", patterns)
    
    # Find all matches in the document
    matches = matcher(doc)

    # Extract the matched phrases and add to soft skills if they contain keywords
    for match_id, start, end in matches:
        span = doc[start:end].text
        # Filter for phrases that are not too long and contain common soft skill keywords.
        if len(span.split()) <= 3:
            if any(word in span.lower() for word in ['skill', 'ability', 'solving', 'management']):
                soft_skills.add(span.lower())
            
    # --- Method 3: Named Entity Recognition (NER) for Technical Skills ---
    # Loop through the named entities found by spaCy
    for ent in doc.ents:
        # If the entity is a PRODUCT or an ORG (often technical tools or companies),
        # add it to technical skills.
        if ent.label_ in ["PRODUCT", "ORG"]:
            technical_skills.add(ent.text)

    # --- Method 4: Explicit Keyword Matching for Specific Skills ---
    # This acts as a fallback or a way to ensure specific, hard-to-capture phrases are included.
    if 'analytical' in resume_text.lower():
        soft_skills.add('analytical')
    if 'problem-solving' in resume_text.lower():
        soft_skills.add('problem-solving')
    if 'Machine Learning' in resume_text:
        technical_skills.add('Machine Learning')
    if 'Deep Learning' in resume_text:
        technical_skills.add('Deep Learning')

    # Combine all found skills into a single set to remove duplicates.
    all_skills_set = technical_skills.union(soft_skills)

    # Return a dictionary with sorted lists for readability.
    return {
        'technical_skills': sorted(list(technical_skills)),
        'soft_skills': sorted(list(soft_skills)),
        'all_skills': sorted(list(all_skills_set))
    }

def generate_skill_report(skills_dict):
    """
    Generates and prints a formatted report of the extracted skills.

    Args:
        skills_dict (dict): The dictionary returned by extract_all_skills.
    """
    print("=== SKILL EXTRACTION REPORT ===")
    
    # Get the lists of skills, defaulting to an empty list if not found.
    tech_skills = skills_dict.get('technical_skills', [])
    s_skills = skills_dict.get('soft_skills', [])

    # Print the technical skills and their count.
    print(f"TECHNICAL SKILLS ({len(tech_skills)}):")
    for skill in tech_skills:
        print(f"  • {skill}")

    # Print the soft skills and their count.
    print(f"SOFT SKILLS ({len(s_skills)}):")
    for skill in s_skills:
        print(f"  • {skill}")

    # Calculate and print the summary statistics.
    total_skills = len(tech_skills) + len(s_skills)
    tech_percentage = (len(tech_skills) / total_skills * 100) if total_skills > 0 else 0
    soft_percentage = (len(s_skills) / total_skills * 100) if total_skills > 0 else 0

    print("\nSUMMARY:")
    print(f"  Total Skills: {total_skills}")
    print(f"  Technical: {len(tech_skills)} ({tech_percentage:.0f}%)")
    print(f"  Soft Skills: {len(s_skills)} ({soft_percentage:.0f}%)")

if __name__ == "__main__":
    # Example resume text for testing the functions.
    resume_input = """
    SKILLS:
    Programming: Python, Java, JavaScript
    Frameworks: TensorFlow, React, Django
    Experience in Machine Learning and Deep Learning
    Strong analytical and problem-solving skills
    """
    
    # Call the main extraction function
    extracted_skills = extract_all_skills(resume_input)
    
    # Print the raw output dictionary
    print("Output of extract_all_skills (Question 5.1):")
    print(extracted_skills)
    
    # Generate and print the formatted report
    generate_skill_report(extracted_skills)