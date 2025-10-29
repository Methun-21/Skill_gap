import os
import re
import sys
import argparse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import docx
from io import BytesIO
import spacy
from spacy.matcher import PhraseMatcher

# --- 1. TEXT EXTRACTION LOGIC (from Task 1) ---

def extract_text_from_file(file_path: str) -> str:
    """Reads a file and extracts its text content based on extension."""
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()
    try:
        with open(file_path, 'rb') as f:
            file_content = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found at: {file_path}")

    if extension == '.pdf':
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
        text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
    elif extension == '.docx':
        doc = docx.Document(BytesIO(file_content))
        text = "\n".join(para.text for para in doc.paragraphs)
    elif extension == '.txt':
        text = file_content.decode('utf-8')
    else:
        raise ValueError(f"Unsupported file type: {extension}")
    return text.strip()

# --- 2. SKILL EXTRACTION LOGIC (from Task 2 & NER Trainer) ---

def load_all_skills():
    """Loads skills from skills_dict.txt and the hardcoded lists from ner_trainer.py."""
    # Skills from ner_trainer.py
    known_skills = {
        "python", "c/c++", "sql", "javascript", "html", "css",
        "react", "numpy", "pandas", "matplotlib", "seaborn", "sklearn",
        "flask", "data structures", "algorithms", "dbms", "operating systems", "oop",
        "software engineering", "full stack development", "cloud & ai integration", 
        "machine learning", "data engineering", "computer vision", "image processing",
        "aws glue", "redshift", "s3", "athena", "etl workflows", "data pipelines",
        "restful services", "vite", "typescript", "tailwind css", "shadcn ui", "mysql",
        "xgboost", "scikit-learn"
    }
    known_soft_skills = {
        "problem solving", "self-learning", "quick learner", "adaptability", 
        "team collaboration", "deep analyzer", "teamwork", "collaboration", "leadership",
        "communication"
    }

    # Load skills from skills_dict.txt
    try:
        with open('../skills_dict.txt', 'r', encoding='utf-8') as f:
            dict_skills = {line.strip().lower() for line in f if line.strip()}
    except FileNotFoundError:
        print("Warning: skills_dict.txt not found. Using a limited set of skills.")
        dict_skills = set()

    # Combine all skills into a single set for uniqueness
    all_skills = known_skills.union(known_soft_skills).union(dict_skills)
    return list(all_skills)

def extract_skills_with_matcher(text: str, skill_list: list):
    """Uses spaCy's PhraseMatcher to extract a definitive list of skills."""
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model 'en_core_web_sm'...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
        
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(skill) for skill in skill_list]
    matcher.add("SkillPattern", patterns)
    
    doc = nlp(text)
    matches = matcher(doc)
    
    found_skills = {doc[start:end].text for _, start, end in matches}
    return sorted(list(found_skills))

# --- 3. MAIN PIPELINE LOGIC (Task 4 & 5) ---

def main(resume_path, jd_path):
    """Runs the full end-to-end skill gap analysis pipeline."""
    print(f"--- Starting Master Pipeline ---\nResume: {resume_path}\nJob Description: {jd_path}\n")

    # STEP 1: PARSE TEXT
    print("Step 1: Parsing text from files...")
    try:
        resume_text = extract_text_from_file(resume_path)
        jd_text = extract_text_from_file(jd_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    # STEP 2: EXTRACT SKILLS
    print("Step 2: Extracting skills using PhraseMatcher...")
    comprehensive_skill_list = load_all_skills()
    resume_skills = extract_skills_with_matcher(resume_text, comprehensive_skill_list)
    jd_skills = extract_skills_with_matcher(jd_text, comprehensive_skill_list)

    if not resume_skills or not jd_skills:
        print("Error: Could not extract skills from one or both documents. Exiting.")
        sys.exit(1)

    print(f"  - Found {len(resume_skills)} unique skills in resume.")
    print(f"  - Found {len(jd_skills)} unique skills in job description.")

    # STEP 3: MATCH SKILLS (EMBEDDINGS)
    print("Step 3: Matching skills using sentence embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    resume_embeddings = model.encode(resume_skills, convert_to_tensor=True)
    jd_embeddings = model.encode(jd_skills, convert_to_tensor=True)
    cosine_scores = util.cos_sim(jd_embeddings, resume_embeddings)

    # STEP 4: CLASSIFY & GENERATE REPORT
    print("Step 4: Classifying matches and generating report...")
    strong_threshold = 0.7
    partial_threshold = 0.4
    report_data = []

    for i in range(len(jd_skills)):
        jd_skill = jd_skills[i]
        scores = cosine_scores[i].cpu().numpy()
        max_score = np.max(scores)
        best_match_index = np.argmax(scores)
        best_resume_skill = resume_skills[best_match_index]
        
        status, resume_match, final_score = ('Missing', '', np.nan)
        if max_score >= strong_threshold:
            status = 'Strong Match'
            resume_match = best_resume_skill
            final_score = max_score
        elif max_score >= partial_threshold:
            status = 'Partial Match'
            resume_match = best_resume_skill
            final_score = max_score
            
        report_data.append({
            'JD Skill': jd_skill,
            'Resume Match': resume_match,
            'Status': status,
            'Score': final_score
        })

    df_report = pd.DataFrame(report_data)
    df_report['Score'] = df_report['Score'].map('{:.4f}'.format, na_action='ignore')
    report_filename = 'skill_gap_report.csv'
    df_report.to_csv(report_filename, index=False)

    print(f"\n--- Pipeline Complete ---")
    print(f"Final report saved to: '{report_filename}'")
    print("\n--- Report Preview ---")
    print(df_report[['JD Skill', 'Resume Match', 'Status']].to_string())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a master pipeline for skill gap analysis.')
    parser.add_argument('resume_path', type=str, help='Path to the resume file (e.g., resume.docx).')
    parser.add_argument('jd_path', type=str, help='Path to the job description file (e.g., jd.docx).')
    args = parser.parse_args()
    main(args.resume_path, args.jd_path)
