# task_1_parser.py
"""
Task 1: Resume & JD Parsing (Deliverable)

Instructions:
1. Loads three specific files from disk (DOCX Resume, PDF Resume, JD DOCX).
2. Uses TextExtractor and TextCleaner classes defined in the separate complete_pipeline.py.
3. Cleans and normalizes the text.
4. Saves the output to the required .txt files.
"""
import os
import sys
import re
import logging
from io import BytesIO
from typing import Dict, List
from datetime import datetime

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Dependencies Check and Import ---
# This block attempts to import the core logic classes (TextExtractor, TextCleaner)
try:
    # Attempt to import necessary classes from the processing file
    from complete_pipeline import TextExtractor, TextCleaner
except ImportError:
    print("Error: Could not import TextExtractor or TextCleaner.")
    print("Please ensure 'complete_pipeline.py' is in the parent directory and contains these classes.")
    sys.exit(1)

# Configure logging (for extraction/cleaning steps)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger('Parser')

# --- Core Parsing Function ---

def parse_document(file_path: str, doc_type: str, output_name: str) -> bool:
    """Loads file content, runs the full cleaning pipeline, and saves output."""

    try:
        # 1. Load the file content as binary data (required by docx/PyPDF2)
        with open(file_path, 'rb') as f:
            file_content = f.read()

        file_name = os.path.basename(file_path)
        file_format = file_name.split('.')[-1].lower()

        file_info = {'name': file_name, 'content': file_content, 'format': file_format}

        # 2. Run Extraction and Cleaning Pipeline
        extractor = TextExtractor()
        cleaner = TextCleaner()

        extraction = extractor.extract_text(file_info)
        if not extraction['success']:
            raise Exception(f"Extraction failed: {extraction['error']}")

        cleaning = cleaner.clean_text(extraction['text'], doc_type)
        if not cleaning['success']:
            raise Exception(f"Cleaning failed: {cleaning['error']}")

        # 3. Save Output to .txt file
        with open(output_name, 'w', encoding='utf-8') as f:
            f.write(cleaning['cleaned_text'])

        print(f"[+] Parsed '{file_name}' ({doc_type.upper()}) -> Saved as '{output_name}'")
        return True

    except FileNotFoundError:
        print(f"[-] ERROR: Source file not found at path: {file_path}")
        return False
    except Exception as e:
        print(f"[-] FATAL ERROR during processing of {file_path}: {e}")
        return False


# --- Task 1 Execution ---

def run_task_1():
    print("\n" + "="*40)
    print("--- Starting Task 1: Document Parsing ---")
    print("="*40)

    # Define file mappings based on your provided inputs
    files_to_process = [
        # (Input File Path, Document Type, Output File Name)
        ("../Skill_gap/sample_resume.docx", "resume", "resume1_parsed.txt"),
        ("../Skill_gap/sample_resume.pdf", "resume", "resume2_parsed.txt"),
        ("../Skill_gap/AI_ML_Engineer_Job_Description.docx", "job_description", "jd_parsed.txt"),
    ]

    results = []
    for file_path, doc_type, output_name in files_to_process:
        success = parse_document(file_path, doc_type, output_name)
        results.append(success)

    if all(results):
        print("\n[SUCCESS] Task 1 Completed Successfully. Deliverables are saved in the current directory.")
    else:
        print("\n[ERROR] Task 1 completed with errors. Check log messages above.")


if __name__ == '__main__':
    run_task_1()
