from sentence_transformers import SentenceTransformer, util
import numpy as np
import re
import os
import pandas as pd

def extract_resume_skills(text):
    """Extracts technical and soft skills from the resume text."""
    skills = []
    skills_section = re.search(r'=== SKILLS ===\n(.*?)(?=\n\n===|\Z)', text, re.DOTALL | re.IGNORECASE)
    if skills_section:
        skills_text = skills_section.group(1)
        headers = ['Languages:', 'Libraries/Tools:', 'Frameworks:', 'Coursework:', 'Interests:', 'Soft Skills:']
        for header in headers:
            skills_text = skills_text.replace(header, ',')
        skills_text = skills_text.replace('\n', ',')
        raw_skills = skills_text.split(',')
        for skill in raw_skills:
            skill = skill.strip()
            if not skill:
                continue
            if 'C/C++' in skill:
                skills.append('C')
                skills.append('C++')
                skill = skill.replace('C/C++', '').strip()
            if '&' in skill:
                skills.extend([s.strip() for s in skill.split('&') if s.strip()])
            elif skill:
                skills.append(skill)
    return list(set([s for s in skills if s and len(s) > 1]))

def extract_jd_skills(text):
    """Extracts skills from the job description text."""
    skills = []
    skill_keywords = [
        'Python', 'PyTorch', 'TensorFlow', 'Scikit-learn', 'Pandas', 'NumPy',
        'machine learning', 'data processing', 'algorithms', 'data structures',
        'cloud platforms', 'AWS', 'GCP', 'Azure',
    ]
    for line in text.split('\n'):
        line = line.strip()
        for keyword in skill_keywords:
            if keyword.lower() in line.lower() and keyword not in skills:
                skills.append(keyword)
    return list(set(skills))

# --- Main Script ---

# 1. Set up cache
model_cache_path = '../model_cache'
if not os.path.exists(model_cache_path):
    os.makedirs(model_cache_path)
os.environ['SENTENCE_TRANSFORMERS_HOME'] = model_cache_path

# 2. Load texts
try:
    with open('resume1_parsed.txt', 'r', encoding='utf-8') as f:
        resume_text = f.read()
except FileNotFoundError:
    print("Error: resume1_parsed.txt not found.")
    resume_text = ""

try:
    with open('jd_parsed.txt', 'r', encoding='utf-8') as f:
        jd_text = f.read()
except FileNotFoundError:
    print("Error: jd_parsed.txt not found.")
    jd_text = ""

# 3. Extract skills
resume_skills = sorted(extract_resume_skills(resume_text))
jd_skills = sorted(extract_jd_skills(jd_text))

if not resume_skills or not jd_skills:
    print("Could not extract skills from resume or job description. Exiting.")
else:
    print("--- Extracted Resume Skills ---")
    print(resume_skills)
    print("\n--- Extracted JD Skills ---")
    print(jd_skills)
    print("\n" + "="*40 + "\n")

    # 4. Embed skills
    print("Loading Sentence-BERT model...")
    model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=model_cache_path)
    
    print("Embedding skills...")
    resume_embeddings = model.encode(resume_skills, convert_to_tensor=True)
    jd_embeddings = model.encode(jd_skills, convert_to_tensor=True)

    # 5. Compute cosine similarity
    cosine_scores = util.cos_sim(resume_embeddings, jd_embeddings)

    # 6. Save the full similarity matrix to a CSV file
    df_matrix = pd.DataFrame(cosine_scores.cpu().numpy(), index=resume_skills, columns=jd_skills)
    matrix_filename = 'similarity_matrix.csv'
    df_matrix.to_csv(matrix_filename)
    print(f"Successfully created '{matrix_filename}'")

    # 7. Generate and save the top-3 matches to a CSV file, with a threshold
    top_matches_data = []
    threshold = 0.3  # Set a minimum similarity threshold

    for i in range(len(resume_skills)):
        row = {'Resume Skill': resume_skills[i]}
        sim_scores = cosine_scores[i].cpu().numpy()
        
        # Get indices of scores above the threshold, then sort them
        strong_match_indices = np.where(sim_scores > threshold)[0]
        sorted_strong_matches = sorted(strong_match_indices, key=lambda idx: sim_scores[idx], reverse=True)

        for j in range(3):
            if j < len(sorted_strong_matches):
                index = sorted_strong_matches[j]
                row[f'JD Match {j+1}'] = jd_skills[index]
                row[f'Score {j+1}'] = sim_scores[index]
            else:
                # Fill with empty values if there are fewer than 3 strong matches
                row[f'JD Match {j+1}'] = ''
                row[f'Score {j+1}'] = np.nan

        top_matches_data.append(row)

    df_top_matches = pd.DataFrame(top_matches_data)
    matches_filename = 'top_3_matches.csv'
    df_top_matches.to_csv(matches_filename, index=False)
    print(f"Successfully created '{matches_filename}'")

    print("\nProcessing complete.")