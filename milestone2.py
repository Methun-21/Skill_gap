"""
COMPLETE Milestone 2: AI-Powered Skill Gap Analyzer - Skill Extraction
Integrated implementation with ALL features:
- Multi-method skill extraction (spaCy, POS, context, NER)
- BERT embeddings with Sentence-BERT
- Custom NER training capability
- Annotation interface for training data
- Comprehensive visualizations
- Multiple export formats
"""

import streamlit as st
import spacy
from spacy.training import Example
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
import subprocess
import re
import json
import random
import logging
from typing import Dict, List, Set, Tuple, Optional
from collections import Counter, defaultdict
from datetime import datetime
from io import BytesIO

# Configure page
st.set_page_config(
    page_title="AI Skill Gap Analyzer - Milestone 2",
    page_icon="🧠",
    layout="wide"
)

# --- NEW COLORFUL CUSTOM CSS ---
st.markdown("""
    <style>
    /* --- Main App Styling --- */
    :root {
        --primary-color: #1e3a8a; /* Deep Blue */
        --secondary-color: #4338ca; /* Indigo */
        --accent-color: #0d9488; /* Teal */
        --text-color: #e2e8f0; /* Light Gray for text */
        --bg-color: #0f172a; /* Slate background */
        --card-bg-color: #1e293b; /* Darker Slate for cards */
    }

    /* Set a dark theme */
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-color);
    }

    /* --- Title and Headers --- */
    h1 {
        color: #f8fafc; /* White */
        background: -webkit-linear-gradient(45deg, #38bdf8, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    h2, h3 {
        color: #93c5fd; /* Light Blue */
    }

    /* --- Buttons --- */
    .stButton > button {
        border-radius: 20px;
        background-image: linear-gradient(to right, var(--accent-color), #14b8a6);
        color: white;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px 0 rgba(13, 148, 136, 0.4);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(13, 148, 136, 0.6);
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* --- Tabs --- */
    button[data-baseweb="tab"][aria-selected="true"] {
        background-image: linear-gradient(to right, var(--secondary-color), #6d28d9);
        color: white;
        border-radius: 8px 8px 0 0;
    }
    
    /* --- Skill Tags --- */
    .skill-tag {
        display: inline-block;
        padding: 6px 12px;
        margin: 4px;
        border-radius: 16px;
        font-size: 14px;
        font-weight: 500;
        border: 1px solid transparent;
        transition: all 0.2s ease-in-out;
    }
    .skill-tag:hover {
        transform: scale(1.05);
    }
    .tech-skill {
        background-color: rgba(59, 130, 246, 0.2); /* Blue bg */
        border-color: #3b82f6;
        color: #93c5fd; /* Light blue text */
    }
    .soft-skill {
        background-color: rgba(168, 85, 247, 0.2); /* Purple bg */
        border-color: #a855f7;
        color: #d8b4fe; /* Light purple text */
    }

    /* --- Containers and Cards --- */
    [data-testid="stExpander"] {
        background-color: var(--card-bg-color);
        border-radius: 10px;
        border: 1px solid #334155;
    }

    </style>
""", unsafe_allow_html=True)


class SkillDatabase:
    """Comprehensive skill database with categorization"""
    
    def __init__(self):
        self.skills = self._initialize_skill_database()
        self.abbreviations = self._initialize_abbreviations()
        self.skill_patterns = self._initialize_skill_patterns()
        self.non_skill_words = {
            'role', 'responsibilities', 'company', 'experience', 'summary',
            'education', 'requirements', 'skills', 'preferred', 'duties',
            'about', 'team', 'projects', 'work', 'contact', 'email', 'phone'
        }
    
    def _initialize_skill_database(self) -> Dict[str, List[str]]:
        """Initialize comprehensive skill database"""
        return {
            'programming_languages': [
                'Python', 'Java', 'JavaScript', 'TypeScript', 'C++', 'C#', 'C',
                'Ruby', 'PHP', 'Swift', 'Kotlin', 'Go', 'Rust', 'Scala', 'R',
                'MATLAB', 'Perl', 'Dart', 'Shell', 'Bash', 'PowerShell'
            ],
            'web_frameworks': [
                'React', 'Angular', 'Vue.js', 'Node.js', 'Express.js', 'Django',
                'Flask', 'FastAPI', 'Spring Boot', 'Spring', 'ASP.NET', '.NET Core',
                'Ruby on Rails', 'Laravel', 'Next.js', 'Nuxt.js', 'Svelte',
                'jQuery', 'Bootstrap', 'Tailwind CSS', 'Material-UI'
            ],
            'databases': [
                'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'Cassandra', 'Oracle',
                'SQL Server', 'SQLite', 'MariaDB', 'DynamoDB', 'Elasticsearch',
                'Firebase', 'Neo4j', 'Snowflake', 'BigQuery'
            ],
            'ml_ai': [
                'Machine Learning', 'Deep Learning', 'Neural Networks',
                'Natural Language Processing', 'NLP', 'Computer Vision',
                'Reinforcement Learning', 'Transfer Learning',
                'Feature Engineering', 'MLOps', 'Generative AI',
                'Large Language Models', 'LLM', 'CNN', 'RNN', 'LSTM',
                'Transformer', 'BERT', 'GPT'
            ],
            'ml_frameworks': [
                'TensorFlow', 'PyTorch', 'Keras', 'Scikit-learn', 'XGBoost',
                'Pandas', 'NumPy', 'SciPy', 'Matplotlib', 'Seaborn',
                'Plotly', 'NLTK', 'spaCy', 'Hugging Face', 'OpenCV'
            ],
            'cloud_platforms': [
                'AWS', 'Amazon Web Services', 'Azure', 'Microsoft Azure',
                'Google Cloud Platform', 'GCP', 'Heroku', 'DigitalOcean'
            ],
            'devops_tools': [
                'Docker', 'Kubernetes', 'Jenkins', 'GitLab CI', 'GitHub Actions',
                'CircleCI', 'Ansible', 'Terraform', 'Prometheus', 'Grafana',
                'ELK Stack', 'Datadog'
            ],
            'version_control': [
                'Git', 'GitHub', 'GitLab', 'Bitbucket', 'SVN'
            ],
            'testing': [
                'Jest', 'Mocha', 'Pytest', 'JUnit', 'Selenium', 'Cypress',
                'Postman', 'JMeter'
            ],
            'soft_skills': [
                'Leadership', 'Team Management', 'Communication',
                'Problem Solving', 'Critical Thinking', 'Analytical Skills',
                'Project Management', 'Collaboration', 'Teamwork',
                'Adaptability', 'Creativity', 'Time Management'
            ]
        }
    
    def _initialize_abbreviations(self) -> Dict[str, str]:
        """Initialize common abbreviations"""
        return {
            'ML': 'Machine Learning', 'DL': 'Deep Learning',
            'AI': 'Artificial Intelligence', 'NLP': 'Natural Language Processing',
            'CV': 'Computer Vision', 'NN': 'Neural Networks',
            'CNN': 'Convolutional Neural Networks', 'RNN': 'Recurrent Neural Networks',
            'K8s': 'Kubernetes', 'K8S': 'Kubernetes',
            'CI/CD': 'Continuous Integration/Continuous Deployment',
            'API': 'Application Programming Interface',
            'REST': 'Representational State Transfer',
            "OS": "Operating System",
            "DB": "Database",
            "JS": "JavaScript",
            "HR": "Human Resources",
            'SQL': 'Structured Query Language', 'OOP': 'Object-Oriented Programming',
            'TDD': 'Test-Driven Development', 'AWS': 'Amazon Web Services',
            'GCP': 'Google Cloud Platform'
        }
    
    def _initialize_skill_patterns(self) -> List[str]:
        """
        Initialize regex patterns for skill detection.

        These patterns are designed to match common phrases in resumes and job descriptions
        that indicate skills, such as 'experience in Python', 'proficient with Django', or
        '5+ years of experience in Machine Learning'. The patterns aim to cover a wide range
        of phrasings to robustly extract both technical and soft skills from varied text sources.
        """
        return [
            r'experience (?:in|with) ([\w\s\+\#\.\-]+)',
            r'proficient (?:in|with|at) ([\w\s\+\#\.\-]+)',
            r'expertise (?:in|with) ([\w\s\+\#\.\-]+)',
            r'knowledge of ([\w\s\+\#\.\-]+)',
            r'skilled (?:in|at|with) ([\w\s\+\#\.\-]+)',
            r'familiar with ([\w\s\+\#\.\-]+)',
            r'(\d+)\+?\s*years? of (?:experience )?(?:in|with) ([\w\s\+\#\.\-]+)'
        ]
    
    def get_all_skills(self) -> List[str]:
        """Get flattened list of all skills"""
        all_skills = []
        for skills in self.skills.values():
            all_skills.extend(skills)
        return all_skills
    
    def get_category_for_skill(self, skill: str) -> Optional[str]:
        """Find which category a skill belongs to"""
        skill_lower = skill.lower()
        for category, skills in self.skills.items():
            if any(s.lower() == skill_lower for s in skills):
                return category
        return 'other'


class TextPreprocessor:
    """Advanced text preprocessing for skill extraction"""
    
    def __init__(self):
        self.nlp = self._load_spacy_model()
        self._customize_stop_words()
    
    def _load_spacy_model(self):
        """Load spaCy model with error handling"""
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            st.warning("⚠️ Installing spaCy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            return spacy.load("en_core_web_sm")
    
    def _customize_stop_words(self):
        """Customize stop words for skill extraction"""
        programming_langs = {'c', 'r', 'go', 'd', 'f'}
        for lang in programming_langs:
            self.nlp.Defaults.stop_words.discard(lang)
    
    def preprocess(self, text: str) -> Dict:
        """Complete preprocessing pipeline"""
        if not text or not text.strip():
            return {'success': False, 'error': 'Empty text'}
        
        try:
            doc = self.nlp(text)
            
            return {
                'success': True,
                'doc': doc,
                'noun_chunks': [chunk.text for chunk in doc.noun_chunks],
                'entities': [(ent.text, ent.label_) for ent in doc.ents],
                'sentences': [sent.text for sent in doc.sents]
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}


class SkillExtractor:
    """Main skill extraction engine"""
    
    def __init__(self):
        self.skill_db = SkillDatabase()
        self.preprocessor = TextPreprocessor()
        self.logger = self._setup_logger()
    
    def extract_skills(self, text: str, document_type: str = 'resume') -> Dict:
        """Extract skills using multiple methods"""
        try:
            preprocess_result = self.preprocessor.preprocess(text)
            if not preprocess_result['success']:
                return {'success': False, 'error': preprocess_result['error']}
            
            doc = preprocess_result['doc']
            
            # Multiple extraction methods
            keyword_skills = self._extract_by_keywords(text)
            pos_skills = self._extract_by_pos_patterns(doc)
            context_skills = self._extract_by_context(text)
            ner_skills = self._extract_by_ner(preprocess_result['entities'])
            chunk_skills = self._extract_from_noun_chunks(preprocess_result['noun_chunks'])
            
            # Combine and deduplicate
            all_skills = self._combine_and_deduplicate([
                keyword_skills, pos_skills, context_skills, ner_skills, chunk_skills
            ])
            
            # Normalize (expand abbreviations)
            normalized_skills = self._normalize_skills(all_skills)
            
            # Categorize skills
            categorized_skills = self._categorize_skills(normalized_skills)
            
            # Calculate confidence scores
            skill_confidence = self._calculate_confidence(
                normalized_skills,
                [keyword_skills, pos_skills, context_skills, ner_skills, chunk_skills]
            )
            
            return {
                'success': True,
                'all_skills': normalized_skills,
                'categorized_skills': categorized_skills,
                'skill_confidence': skill_confidence,
                'extraction_methods': {
                    'keyword_matching': len(keyword_skills),
                    'pos_patterns': len(pos_skills),
                    'context_based': len(context_skills),
                    'ner': len(ner_skills),
                    'noun_chunks': len(chunk_skills)
                },
                'statistics': {
                    'total_skills': len(normalized_skills),
                    'technical_skills': sum(len(skills) for cat, skills in categorized_skills.items() if cat != 'soft_skills'),
                    'soft_skills': len(categorized_skills.get('soft_skills', []))
                }
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _extract_by_keywords(self, text: str) -> Set[str]:
        """Extract skills by keyword matching"""
        found_skills = set()
        text_lower = text.lower()
        
        for skill in self.skill_db.get_all_skills():
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.add(skill)
        
        return found_skills
    
    def _extract_by_pos_patterns(self, doc) -> Set[str]:
        """Extract skills using POS patterns"""
        found_skills = set()
        tokens = list(doc)
        
        # ADJ + NOUN patterns
        for i in range(len(tokens) - 1):
            if tokens[i].pos_ == 'ADJ' and tokens[i+1].pos_ in ['NOUN', 'PROPN']:
                pattern = f"{tokens[i].text} {tokens[i+1].text}"
                if self._is_valid_skill(pattern):
                    found_skills.add(pattern)
        
        # Proper nouns
        for token in doc:
            if token.pos_ == 'PROPN' and self._is_valid_skill(token.text):
                found_skills.add(token.text)
        
        return found_skills
    
    def _extract_by_context(self, text: str) -> Set[str]:
        """Extract skills based on context patterns"""
        found_skills = set()
        
        for pattern in self.skill_db.skill_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 1:
                    skill_text = match.group(len(match.groups()))
                    skills = self._clean_and_split_skills(skill_text)
                    for skill in skills:
                        if self._is_valid_skill(skill):
                            found_skills.add(skill)
        
        return found_skills
    
    def _extract_by_ner(self, entities: List[Tuple[str, str]]) -> Set[str]:
        """Extract skills from named entities"""
        found_skills = set()
        relevant_labels = ['ORG', 'PRODUCT', 'GPE']
        
        for entity_text, label in entities:
            if label in relevant_labels and self._is_valid_skill(entity_text):
                found_skills.add(entity_text)
        
        return found_skills
    
    def _extract_from_noun_chunks(self, noun_chunks: List[str]) -> Set[str]:
        """Extract skills from noun chunks"""
        found_skills = set()
        
        for chunk in noun_chunks:
            chunk_clean = chunk.strip()
            if self._is_valid_skill(chunk_clean):
                found_skills.add(chunk_clean)
        
        return found_skills
    
    def _is_valid_skill(self, text: str) -> bool:
        """Validate if text is a valid skill"""
        text_clean = text.strip()
        text_lower = text_clean.lower()

        if text_lower in self.skill_db.non_skill_words:
            return False

        if not text_clean or len(text_clean) < 2:
            return False
    
        all_skills_lower = [s.lower() for s in self.skill_db.get_all_skills()]
    
        if text_lower in all_skills_lower:
            return True
    
        for skill in self.skill_db.get_all_skills():
            if skill.lower() in text_lower or text_lower in skill.lower():
                if abs(len(skill) - len(text_clean)) <= 3:
                    return True
    
        return False    
    
    def _clean_and_split_skills(self, text: str) -> List[str]:
        """Clean and split comma-separated skills"""
        skills = re.split(r'[,;|/&]|\band\b', text)
        cleaned_skills = []
        
        for skill in skills:
            skill_clean = skill.strip()
            skill_clean = re.sub(r'\b(etc|and more)\b', '', skill_clean, flags=re.IGNORECASE).strip()
            if skill_clean and len(skill_clean) > 1:
                cleaned_skills.append(skill_clean)
        
        return cleaned_skills
    
    def _combine_and_deduplicate(self, skill_sets: List[Set[str]]) -> List[str]:
        """Combine and remove duplicates"""
        combined = set()
        for skill_set in skill_sets:
            combined.update(skill_set)
        
        unique_skills = {}
        for skill in combined:
            skill_lower = skill.lower()
            if skill_lower not in unique_skills:
                unique_skills[skill_lower] = skill
        
        return sorted(unique_skills.values())
    
    def _normalize_skills(self, skills: List[str]) -> List[str]:
        """Normalize skill names (expand abbreviations)"""
        normalized = []
        
        for skill in skills:
            if skill.upper() in self.skill_db.abbreviations:
                normalized.append(self.skill_db.abbreviations[skill.upper()])
            else:
                normalized.append(skill)
        
        return sorted(set(normalized))
    
    def _categorize_skills(self, skills: List[str]) -> Dict[str, List[str]]:
        """Categorize skills"""
        categorized = defaultdict(list)
        
        for skill in skills:
            category = self.skill_db.get_category_for_skill(skill)
            categorized[category].append(skill)
        
        for category in categorized:
            categorized[category] = sorted(categorized[category])
        
        return dict(categorized)
    
    def _calculate_confidence(self, skills: List[str], method_results: List[Set[str]]) -> Dict[str, float]:
        """Calculate confidence score for each skill"""
        confidence_scores = {}
        
        for skill in skills:
            detection_count = sum(
                1 for method_set in method_results 
                if skill in method_set or skill.lower() in {s.lower() for s in method_set}
            )
            confidence = detection_count / len(method_results)
            confidence_scores[skill] = round(confidence, 2)
        
        return confidence_scores
    
    def _setup_logger(self):
        """Setup logging"""
        logger = logging.getLogger('SkillExtractor')
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger


class SentenceBERTEmbedder:
    """Generate and manage BERT embeddings for skills"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize Sentence-BERT model"""
        try:
            # --- FIX: Added device='cpu' to the constructor ---
            self.model = SentenceTransformer(model_name, device='cpu')
            self.skill_embeddings = {}
        except Exception as e:
            st.error(f"❌ Failed to load BERT model: {e}")
            st.info("Attempting to install sentence-transformers...")
            try:
                subprocess.run(["pip", "install", "sentence-transformers"])
                # Retry loading the model after installation
                self.model = SentenceTransformer(model_name, device='cpu')
                st.success("✅ sentence-transformers installed and model loaded successfully!")
            except Exception as install_e:
                st.error(f"Failed to auto-install or load model after installation: {install_e}")
                self.model = None

    def encode_skills(self, skills: List[str]) -> Dict[str, np.ndarray]:
        """Generate embeddings for skills"""
        if not skills or self.model is None:
            return {}
        
        embeddings = self.model.encode(skills, show_progress_bar=True)
        
        skill_embeddings = {}
        for skill, embedding in zip(skills, embeddings):
            skill_embeddings[skill] = embedding
            self.skill_embeddings[skill] = embedding
        
        return skill_embeddings
    
    def compute_similarity(self, skill1: str, skill2: str) -> float:
        """Compute cosine similarity between two skills"""
        if self.model is None: return 0.0

        if skill1 not in self.skill_embeddings:
            emb1 = self.model.encode([skill1])[0]
            self.skill_embeddings[skill1] = emb1
        else:
            emb1 = self.skill_embeddings[skill1]
        
        if skill2 not in self.skill_embeddings:
            emb2 = self.model.encode([skill2])[0]
            self.skill_embeddings[skill2] = emb2
        else:
            emb2 = self.skill_embeddings[skill2]
        
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        return float(similarity)
    
    def compute_similarity_matrix(self, skills1: List[str], skills2: List[str]) -> np.ndarray:
        """Compute similarity matrix between two skill sets"""
        if self.model is None: return np.array([])
        embeddings1 = self.model.encode(skills1)
        embeddings2 = self.model.encode(skills2)
        return cosine_similarity(embeddings1, embeddings2)
    
    def find_similar_skills(self, target_skill: str, skill_list: List[str], 
                            threshold: float = 0.7, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find skills similar to target skill"""
        if self.model is None: return []
        similarities = []
        
        for skill in skill_list:
            if skill.lower() != target_skill.lower():
                sim = self.compute_similarity(target_skill, skill)
                if sim >= threshold:
                    similarities.append((skill, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class CustomNERTrainer:
    """Train custom NER model for skill detection"""
    
    def __init__(self):
        self.nlp = None
        self.ner = None
    
    def prepare_training_data(self, annotations: List[Dict]) -> List[Tuple]:
        """Convert annotations to spaCy format"""
        training_data = []
        
        for annotation in annotations:
            text = annotation['text']
            entities = []
            
            for skill in annotation['skills']:
                entities.append((skill['start'], skill['end'], skill['label']))
            
            training_data.append((text, {"entities": entities}))
        
        return training_data
    
    def create_blank_model(self):
        """Create blank spaCy model"""
        self.nlp = spacy.blank("en")
        
        if "ner" not in self.nlp.pipe_names:
            self.ner = self.nlp.add_pipe("ner")
        else:
            self.ner = self.nlp.get_pipe("ner")
        
        self.ner.add_label("SKILL")
    
    def train(self, training_data: List[Tuple], n_iterations: int = 30) -> Dict:
        """Train the NER model"""
        if not self.nlp:
            self.create_blank_model()
        
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "ner"]
        training_stats = {'losses': [], 'iterations': n_iterations}
        
        with self.nlp.disable_pipes(*other_pipes):
            optimizer = self.nlp.begin_training()
            
            for iteration in range(n_iterations):
                random.shuffle(training_data)
                losses = {}
                
                for text, annotations in training_data:
                    doc = self.nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    self.nlp.update([example], drop=0.5, losses=losses)
                
                training_stats['losses'].append(losses.get('ner', 0))
        
        return training_stats
    
    def predict(self, text: str) -> List[Tuple[str, int, int]]:
        """Predict skills in text"""
        if not self.nlp:
            raise ValueError("Model not trained")
        
        doc = self.nlp(text)
        return [(ent.text, ent.start_char, ent.end_char) for ent in doc.ents if ent.label_ == "SKILL"]


class AnnotationInterface:
    """Interface for creating NER training data"""
    
    def __init__(self):
        if 'training_annotations' not in st.session_state:
            st.session_state.training_annotations = []
        if 'current_skills' not in st.session_state:
            st.session_state.current_skills = []
    
    def create_annotation_ui(self):
        """Create annotation UI"""
        st.header("🏷️ Create NER Training Data")
        
        st.markdown("""
        **Instructions:**
        1. Enter text containing skills
        2. Mark skill positions (start/end character indices)
        3. Add to training dataset
        4. Export for model training
        """)
        
        input_text = st.text_area(
            "Enter text to annotate:",
            height=150,
            placeholder="Example: I am a Python developer with 5 years of Machine Learning experience."
        )
        
        if input_text:
            st.markdown("---")
            st.text(input_text)
            
            with st.form("skill_annotation_form"):
                st.markdown("**Add Skill Annotation:**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    skill_text = st.text_input("Skill text")
                with col2:
                    start_pos = st.number_input("Start position", min_value=0, value=0)
                with col3:
                    end_pos = st.number_input("End position", min_value=0, value=0)
                
                if skill_text and start_pos < end_pos:
                    extracted = input_text[start_pos:end_pos]
                    if extracted.strip():
                        st.info(f"Preview: '{extracted}'")
                
                submitted = st.form_submit_button("➕ Add Skill")
                
                if submitted and skill_text and start_pos < end_pos:
                    st.session_state.current_skills.append({
                        'text': skill_text,
                        'start': start_pos,
                        'end': end_pos,
                        'label': 'SKILL'
                    })
                    st.success(f"✅ Added: {skill_text}")
                    st.rerun()
            
            if st.session_state.current_skills:
                st.markdown("**Skills in current text:**")
                for i, skill in enumerate(st.session_state.current_skills):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"{i+1}. **{skill['text']}** ({skill['start']}-{skill['end']})")
                    with col2:
                        if st.button("🗑️", key=f"remove_{i}"):
                            st.session_state.current_skills.pop(i)
                            st.rerun()
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Save Annotation", type="primary"):
                    if st.session_state.current_skills:
                        annotation = {
                            'text': input_text,
                            'skills': st.session_state.current_skills.copy(),
                            'timestamp': datetime.now().isoformat()
                        }
                        st.session_state.training_annotations.append(annotation)
                        st.session_state.current_skills = []
                        st.success(f"✅ Saved! Total: {len(st.session_state.training_annotations)}")
                        st.rerun()
            
            with col2:
                if st.button("🔄 Clear Current"):
                    st.session_state.current_skills = []
                    st.rerun()
        
        if st.session_state.training_annotations:
            st.markdown("---")
            st.subheader(f"📚 Training Dataset ({len(st.session_state.training_annotations)} annotations)")
            
            for i, annotation in enumerate(st.session_state.training_annotations):
                with st.expander(f"Annotation {i+1}: {len(annotation['skills'])} skills"):
                    st.text(annotation['text'])
                    st.write("**Skills:**")
                    for skill in annotation['skills']:
                        st.write(f"- {skill['text']} ({skill['start']}-{skill['end']})")
            
            col1, col2 = st.columns(2)
            
            with col1:
                training_json = json.dumps(st.session_state.training_annotations, indent=2)
                st.download_button(
                    "📥 Download Training Data (JSON)",
                    training_json,
                    f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )
            
            with col2:
                trainer = CustomNERTrainer()
                spacy_format = trainer.prepare_training_data(st.session_state.training_annotations)
                spacy_json = json.dumps(spacy_format, indent=2)
                
                st.download_button(
                    "📥 Download spaCy Format",
                    spacy_json,
                    f"spacy_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )


class SkillVisualizer:
    """Visualize extracted skills"""
    
    @staticmethod
    def create_category_distribution_chart(categorized_skills: Dict[str, List[str]]) -> go.Figure:
        """Create pie chart for category distribution"""
        category_names = {
            'programming_languages': 'Programming Languages',
            'web_frameworks': 'Web Frameworks',
            'databases': 'Databases',
            'ml_ai': 'ML/AI',
            'ml_frameworks': 'ML Frameworks',
            'cloud_platforms': 'Cloud Platforms',
            'devops_tools': 'DevOps Tools',
            'version_control': 'Version Control',
            'testing': 'Testing',
            'soft_skills': 'Soft Skills',
            'other': 'Other'
        }
        
        categories = []
        counts = []
        
        for category, skills in categorized_skills.items():
            if skills:
                categories.append(category_names.get(category, category.replace('_', ' ').title()))
                counts.append(len(skills))
        
        fig = go.Figure(data=[go.Pie(
            labels=categories,
            values=counts,
            hole=0.3,
            textposition='auto',
            textinfo='label+percent+value'
        )])
        
        fig.update_layout(title="Skill Distribution by Category", height=500)
        return fig
    
    @staticmethod
    def create_top_skills_chart(skills: List[str], confidence_scores: Dict[str, float], top_n: int = 15) -> go.Figure:
        """Create bar chart for top skills"""
        sorted_skills = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        skill_names = [skill for skill, _ in sorted_skills]
        confidences = [score * 100 for _, score in sorted_skills]
        
        fig = go.Figure(data=[go.Bar(
            x=confidences,
            y=skill_names,
            orientation='h',
            marker=dict(
                color=confidences,
                colorscale='Viridis',
                colorbar=dict(title="Confidence %")
            ),
            text=[f"{conf:.0f}%" for conf in confidences],
            textposition='auto'
        )])
        
        fig.update_layout(
            title=f"Top {top_n} Skills by Confidence Score",
            xaxis_title="Confidence Score (%)",
            yaxis_title="Skills",
            height=600,
            yaxis=dict(autorange="reversed")
        )
        
        return fig
    
    @staticmethod
    def create_extraction_methods_chart(extraction_methods: Dict[str, int]) -> go.Figure:
        """Create bar chart for extraction methods"""
        method_names = {
            'keyword_matching': 'Keyword Matching',
            'pos_patterns': 'POS Patterns',
            'context_based': 'Context-Based',
            'ner': 'Named Entity Recognition',
            'noun_chunks': 'Noun Chunks'
        }
        
        methods = [method_names.get(m, m) for m in extraction_methods.keys()]
        counts = list(extraction_methods.values())
        
        fig = go.Figure(data=[go.Bar(
            x=methods,
            y=counts,
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            text=counts,
            textposition='auto'
        )])
        
        fig.update_layout(
            title="Skills Detected by Each Extraction Method",
            xaxis_title="Extraction Method",
            yaxis_title="Number of Skills Found",
            height=400
        )
        
        return fig


class CompleteSkillExtractionApp:
    """Complete Milestone 2 application"""
    
    def __init__(self):
        self.skill_extractor = SkillExtractor()
        self.visualizer = SkillVisualizer()
        self.bert_embedder = SentenceBERTEmbedder()
        self.ner_trainer = CustomNERTrainer()
        self.annotator = AnnotationInterface()
        
        # Initialize session state
        if 'extraction_results' not in st.session_state:
            st.session_state.extraction_results = None
        if 'skill_embeddings' not in st.session_state:
            st.session_state.skill_embeddings = None
        if 'trained_ner' not in st.session_state:
            st.session_state.trained_ner = None
    
    def run(self):
        """Run the complete application"""
        st.title("🎯 AI Skill Gap Analyzer - Complete Milestone 2")
        st.markdown("### Advanced Skill Extraction with BERT Embeddings & Custom NER")
        
        tabs = st.tabs([
            "📄 Extract Skills",
            "🧠 BERT Embeddings",
            "🏋️ Train Custom NER",
            "🏷️ Annotate Data",
            "📊 Visualizations",
            "📥 Export"
        ])
        
        with tabs[0]:
            self._skill_extraction_tab()
        
        with tabs[1]:
            self._bert_embeddings_tab()
        
        with tabs[2]:
            self._ner_training_tab()
        
        with tabs[3]:
            self._annotation_tab()
        
        with tabs[4]:
            self._visualization_tab()
        
        with tabs[5]:
            self._export_tab()
    
    def _skill_extraction_tab(self):
        """Skill extraction interface"""
        st.header("✨ Extract Skills from Text")
        
        input_method = st.radio(
            "Choose input method:",
            ["Paste Text", "Upload File"],
            horizontal=True
        )
        
        text_input = ""
        doc_type = "resume"
        
        if input_method == "Paste Text":
            col1, col2 = st.columns([3, 1])
            
            with col1:
                text_input = st.text_area(
                    "Paste resume or job description text:",
                    height=300,
                    placeholder="Paste your resume or job description here..."
                )
            
            with col2:
                doc_type = st.selectbox("Document Type:", ["resume", "job_description"])  
        else:
            uploaded_file = st.file_uploader("Upload document", type=['txt'])
            doc_type = st.selectbox("Document Type:", ["resume", "job_description"])
    
            if uploaded_file:
                try:
                    # **(New code block starts here)**
                    # 1. Try standard UTF-8 decoding
                    text_input = uploaded_file.getvalue().decode("utf-8")
                except UnicodeDecodeError:
                    # 2. If it fails, try the common Windows encoding (CP1252)
                    try:
                        text_input = uploaded_file.getvalue().decode("cp1252")
                        st.warning("Successfully uploaded")
                    except UnicodeDecodeError:
                        # 3. Fallback to Latin-1 (lossless, handles all single bytes)
                        text_input = uploaded_file.getvalue().decode("latin-1")
                        st.warning("⚠️ File required decoding with fallback Latin-1.")
        
        if st.button("🔍 Extract Skills", type="primary", use_container_width=True):
            if text_input:
                with st.spinner("Extracting skills..."):
                    result = self.skill_extractor.extract_skills(text_input, doc_type)
                    
                    if result['success']:
                        st.session_state.extraction_results = result
                        st.success(f"✅ Successfully extracted {result['statistics']['total_skills']} skills!")
                        self._display_extraction_results(result)
                    else:
                        st.error(f"❌ Extraction failed: {result.get('error', 'Unknown error')}")
            else:
                st.warning("⚠️ Please provide text input")
    
    def _display_extraction_results(self, result: Dict):
        """Display extraction results"""
        st.subheader("📊 Extraction Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Skills", result['statistics']['total_skills'])
        with col2:
            st.metric("Technical Skills", result['statistics']['technical_skills'])
        with col3:
            st.metric("Soft Skills", result['statistics']['soft_skills'])
        with col4:
            avg_confidence = sum(result['skill_confidence'].values()) / len(result['skill_confidence']) if result['skill_confidence'] else 0
            st.metric("Avg Confidence", f"{avg_confidence:.0%}")
        
        st.subheader("🏷️ Categorized Skills")
        
        categorized = result['categorized_skills']
        category_items = list(categorized.items())
        
        for i in range(0, len(category_items), 2):
            cols = st.columns(2)
            
            for j, col in enumerate(cols):
                if i + j < len(category_items):
                    category, skills = category_items[i + j]
                    
                    with col:
                        category_display = category.replace('_', ' ').title()
                        st.markdown(f"**{category_display}** ({len(skills)})")
                        
                        skill_html = ""
                        for skill in skills[:10]:
                            confidence = result['skill_confidence'].get(skill, 0)
                            color = "tech-skill" if category != "soft_skills" else "soft-skill"
                            skill_html += f'<span class="skill-tag {color}" title="Confidence: {confidence:.0%}">{skill}</span>'
                        
                        if len(skills) > 10:
                            skill_html += f'<span class="skill-tag">+{len(skills) - 10} more</span>'
                        
                        st.markdown(skill_html, unsafe_allow_html=True)
                        st.markdown("")
        
        with st.expander("🔧 Extraction Methods Used"):
            methods_df = pd.DataFrame([
                {'Method': method.replace('_', ' ').title(), 'Skills Found': count}
                for method, count in result['extraction_methods'].items()
            ])
            st.dataframe(methods_df, use_container_width=True)
    
    def _bert_embeddings_tab(self):
        """BERT embeddings interface"""
        st.header("🧠 Skill Embeddings with Sentence-BERT")
        
        st.markdown("""
        **Sentence-BERT** creates semantic embeddings for skills, enabling:
        - Similarity computation between skills
        - Semantic skill matching
        - Finding related skills
        """)
        
        if not st.session_state.extraction_results:
            st.info("👆 Extract skills first to generate embeddings")
            return
        
        result = st.session_state.extraction_results
        skills = result['all_skills']
        
        if st.button("🚀 Generate BERT Embeddings", type="primary"):
            with st.spinner("Generating embeddings..."):
                embeddings = self.bert_embedder.encode_skills(skills)
                st.session_state.skill_embeddings = embeddings
                st.success(f"✅ Generated embeddings for {len(skills)} skills!")
        
        if st.session_state.skill_embeddings:
            st.subheader("🔍 Skill Similarity Calculator")
            
            col1, col2 = st.columns(2)
            
            with col1:
                skill1 = st.selectbox("Select first skill:", skills, key="sim_skill1")
            
            with col2:
                skill2 = st.selectbox("Select second skill:", skills, key="sim_skill2")
            
            if st.button("Calculate Similarity"):
                similarity = self.bert_embedder.compute_similarity(skill1, skill2)
                
                st.metric(
                    "Similarity Score",
                    f"{similarity:.2%}",
                    delta=f"{'High' if similarity > 0.7 else 'Medium' if similarity > 0.4 else 'Low'} similarity"
                )
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=similarity * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Similarity"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#14b8a6"},
                        'steps': [
                            {'range': [0, 40], 'color': "#334155"},
                            {'range': [40, 70], 'color': "#475569"},
                        ]
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("🎯 Find Similar Skills")
            
            target_skill = st.selectbox("Select target skill:", skills, key="target_skill")
            threshold = st.slider("Similarity threshold:", 0.0, 1.0, 0.7, 0.05)
            
            if st.button("Find Similar Skills"):
                similar_skills = self.bert_embedder.find_similar_skills(
                    target_skill,
                    [s for s in skills if s != target_skill],
                    threshold=threshold,
                    top_k=10
                )
                
                if similar_skills:
                    st.success(f"Found {len(similar_skills)} similar skills:")
                    
                    df = pd.DataFrame(similar_skills, columns=['Skill', 'Similarity'])
                    df['Similarity'] = df['Similarity'].apply(lambda x: f"{x:.2%}")
                    st.dataframe(df, use_container_width=True)
                    
                    fig = go.Figure(data=[go.Bar(
                        x=[s[1] for s in similar_skills],
                        y=[s[0] for s in similar_skills],
                        orientation='h',
                        marker_color='lightblue'
                    )])
                    
                    fig.update_layout(
                        title=f"Skills Similar to '{target_skill}'",
                        xaxis_title="Similarity Score",
                        yaxis_title="Skill",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"No skills found with similarity >= {threshold:.0%}")
            
            st.subheader("📊 Skill Similarity Matrix")
            
            if st.button("Generate Similarity Matrix"):
                with st.spinner("Computing similarities..."):
                    if len(skills) > 1:
                        top_skills = skills[:20]
                        similarity_matrix = self.bert_embedder.compute_similarity_matrix(
                            top_skills,
                            top_skills
                        )
                        
                        fig = go.Figure(data=go.Heatmap(
                            z=similarity_matrix,
                            x=top_skills,
                            y=top_skills,
                            colorscale='Teal',
                            text=similarity_matrix,
                            texttemplate='%{text:.2f}',
                            textfont={"size": 8}
                        ))
                        
                        fig.update_layout(
                            title="Skill Similarity Heatmap (Top 20 Skills)",
                            height=700,
                            width=800
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Need at least two skills to generate a matrix.")

    def _ner_training_tab(self):
        """Custom NER training interface"""
        st.header("🏋️ Train Custom NER Model")
        
        st.markdown("""
        Train a custom spaCy NER model to detect skills in text.
        
        **Steps:**
        1. Load training data (use Annotate Data tab)
        2. Configure training parameters
        3. Train the model
        4. Test the model
        """)
        
        st.subheader("1️⃣ Load Training Data")
        
        training_source = st.radio(
            "Training data source:",
            ["Use Annotated Data", "Upload JSON File"],
            horizontal=True
        )
        
        training_data = None
        
        if training_source == "Use Annotated Data":
            if st.session_state.get('training_annotations'):
                st.success(f"✅ {len(st.session_state.training_annotations)} annotations available")
                training_data = self.ner_trainer.prepare_training_data(
                    st.session_state.training_annotations
                )
            else:
                st.warning("⚠️ No annotations found. Use 'Annotate Data' tab first.")
        else:
            uploaded_file = st.file_uploader("Upload training data (JSON)", type=['json'])
            if uploaded_file:
                try:
                    annotations = json.load(uploaded_file)
                    training_data = self.ner_trainer.prepare_training_data(annotations)
                    st.success(f"✅ Loaded {len(training_data)} training examples")
                except Exception as e:
                    st.error(f"❌ Error loading file: {e}")
        
        if training_data:
            st.subheader("2️⃣ Configure Training")
            
            col1, col2 = st.columns(2)
            
            with col1:
                n_iterations = st.number_input(
                    "Number of iterations:",
                    min_value=10,
                    max_value=100,
                    value=30,
                    step=10
                )
            
            with col2:
                st.info(f"Training examples: {len(training_data)}")
            
            st.subheader("3️⃣ Train Model")
            
            if st.button("🚀 Start Training", type="primary"):
                with st.spinner("Training model..."):
                    
                    try:
                        self.ner_trainer.create_blank_model()
                        training_stats = self.ner_trainer.train(training_data, n_iterations=n_iterations)
                        
                        st.session_state.trained_ner = self.ner_trainer
                        st.session_state.training_stats = training_stats
                        
                        st.success("✅ Training complete!")
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=list(range(1, len(training_stats['losses']) + 1)),
                            y=training_stats['losses'],
                            mode='lines+markers',
                            name='Training Loss'
                        ))
                        
                        fig.update_layout(
                            title="Training Loss Over Iterations",
                            xaxis_title="Iteration",
                            yaxis_title="Loss",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Training failed: {e}")
            
            if st.session_state.get('trained_ner'):
                st.subheader("4️⃣ Test Model")
                
                test_text = st.text_area(
                    "Enter text to test model:",
                    placeholder="Example: I am proficient in Python, Java, and Machine Learning."
                )
                
                if st.button("🧪 Test"):
                    if test_text:
                        try:
                            predictions = st.session_state.trained_ner.predict(test_text)
                            
                            if predictions:
                                st.success(f"✅ Found {len(predictions)} skills:")
                                for skill, start, end in predictions:
                                    st.markdown(f"- **{skill}** (position {start}-{end})")
                            else:
                                st.warning("No skills detected")
                        except Exception as e:
                            st.error(f"❌ Prediction failed: {e}")
    
    def _annotation_tab(self):
        """Annotation interface"""
        self.annotator.create_annotation_ui()
    
    def _visualization_tab(self):
        """Visualization interface"""
        if not st.session_state.extraction_results:
            st.info("👆 Please extract skills first in the 'Extract Skills' tab")
            return
        
        result = st.session_state.extraction_results
        
        st.header("📊 Skill Analysis Visualizations")
        
        st.plotly_chart(self.visualizer.create_category_distribution_chart(result['categorized_skills']), use_container_width=True)
        
        st.subheader("Top Skills by Confidence Score")
        top_n = st.slider("Number of top skills to display:", 5, 30, 15)
        st.plotly_chart(self.visualizer.create_top_skills_chart(
            result['all_skills'],
            result['skill_confidence'],
            top_n
        ), use_container_width=True)
        
        st.plotly_chart(self.visualizer.create_extraction_methods_chart(result['extraction_methods']), use_container_width=True)
        
        with st.expander("📋 Detailed Skill Table"):
            detailed_data = []
            for skill in result['all_skills']:
                category = self.skill_extractor.skill_db.get_category_for_skill(skill)
                confidence = result['skill_confidence'].get(skill, 0)
                
                detailed_data.append({
                    'Skill': skill,
                    'Category': category.replace('_', ' ').title(),
                    'Confidence': f"{confidence:.0%}",
                    'Confidence Score': confidence
                })
            
            df = pd.DataFrame(detailed_data)
            
            st.dataframe(
                df.sort_values('Confidence Score', ascending=False),
                use_container_width=True,
                hide_index=True,
                column_config={"Confidence Score": None}
            )
    
    def _export_tab(self):
        """Export interface"""
        if not st.session_state.extraction_results:
            st.info("👆 Please extract skills first in the 'Extract Skills' tab")
            return
        
        result = st.session_state.extraction_results
        
        st.header("📥 Export Extracted Skills")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_data = self._create_csv_export(result)
            st.download_button(
                label="📄 Download CSV",
                data=csv_data,
                file_name=f"extracted_skills_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            json_data = self._create_json_export(result)
            st.download_button(
                label="📋 Download JSON",
                data=json_data,
                file_name=f"extracted_skills_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col3:
            report_data = self._create_text_report(result)
            st.download_button(
                label="📑 Download Report",
                data=report_data,
                file_name=f"skill_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    def _create_csv_export(self, result: Dict) -> str:
        """Create CSV export"""
        data = []
        for skill in result['all_skills']:
            category = self.skill_extractor.skill_db.get_category_for_skill(skill)
            confidence = result['skill_confidence'].get(skill, 0)
            data.append({
                'Skill': skill, 'Category': category, 'Confidence': confidence,
                'Type': 'Soft Skill' if category == 'soft_skills' else 'Technical Skill'
            })
        return pd.DataFrame(data).to_csv(index=False)
    
    def _create_json_export(self, result: Dict) -> str:
        """Create JSON export"""
        export_data = {
            'extraction_timestamp': datetime.now().isoformat(),
            'statistics': result['statistics'],
            'skills': {
                'all_skills': result['all_skills'],
                'categorized_skills': result['categorized_skills'],
                'skill_confidence': result['skill_confidence']
            },
            'extraction_methods': result['extraction_methods']
        }
        return json.dumps(export_data, indent=2)
    
    def _create_text_report(self, result: Dict) -> str:
        """Create formatted text report"""
        report = ["=" * 80, "SKILL EXTRACTION REPORT", "=" * 80]
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.extend([
            f"\nTotal Skills Extracted: {result['statistics']['total_skills']}",
            f"Technical Skills: {result['statistics']['technical_skills']}",
            f"Soft Skills: {result['statistics']['soft_skills']}",
            "\n" + "-" * 80, "\nCATEGORIZED SKILLS", "-" * 80
        ])
        
        for category, skills in sorted(result['categorized_skills'].items()):
            if skills:
                report.append(f"\n{category.replace('_', ' ').title()} ({len(skills)}):")
                for skill in skills:
                    confidence = result['skill_confidence'].get(skill, 0)
                    report.append(f"  • {skill} (Confidence: {confidence:.0%})")
        
        report.extend(["\n" + "=" * 80, "END OF REPORT", "=" * 80])
        return "\n".join(report)


def main():
    """Main application entry point"""
    try:
        app = CompleteSkillExtractionApp()
        app.run()
        
        with st.sidebar:
            st.header("🎯 Milestone 2 Achieved")
            st.markdown("""
            **All major features have been completed:**
            
            1. **🧠 Skill Extraction:**
               - NLP workflow with spaCy
               - Multi-method extraction
               - 5 extraction techniques
            
            2. **🔍BERT Embeddings**
               - BERT & Sentence-BERT implementations
                -Semantic similarity computation
            
            3. **🏗️ Custom NER Training:**
               - Model training interface
               -Testing tools and visualization
            
            4. **📝 Annotation & Data:**
               - Annotation interface
               - Training data creation
               - Export to CSV, JSON, Text
                        
            5. **📊 Visualizations:**
               - Interactive charts
               - Distribution analysis
               - Similarity heatmaps
            """)
            
            if st.session_state.get('extraction_results'):
                result = st.session_state.extraction_results
                st.header("📊 Current Stats")
                st.metric("Skills Found", result['statistics']['total_skills'])
                avg_conf = sum(result['skill_confidence'].values()) / len(result['skill_confidence']) if result['skill_confidence'] else 0
                st.metric("Avg Confidence", f"{avg_conf:.0%}")
    
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()