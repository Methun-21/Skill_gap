import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime
import json
import io
import base64
import logging

# Configure page
st.set_page_config(
    page_title="AI Skill Gap Analyzer - Milestone 3",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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

    /* --- Tabs --- */
    button[data-baseweb="tab"][aria-selected="true"] {
        background-image: linear-gradient(to right, var(--secondary-color), #6d28d9);
        color: white;
        border-radius: 8px 8px 0 0;
    }

    /* --- Containers and Cards --- */
    [data-testid="stExpander"] {
        background-color: var(--card-bg-color);
        border-radius: 10px;
        border: 1px solid #334155;
    }
    [data-testid="stMetric"] {
        background-color: var(--card-bg-color);
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #334155;
    }
    
    /* --- Custom Styles for Milestone 3 Analysis --- */
    .strong-match {
        background-color: rgba(40, 167, 69, 0.2);
        border: 1px solid #28a745;
        color: #81c784;
        padding: 6px 12px;
        border-radius: 16px;
        margin: 4px;
        display: inline-block;
        font-weight: 500;
    }
    .partial-match {
        background-color: rgba(255, 193, 7, 0.2);
        border: 1px solid #ffc107;
        color: #ffd54f;
        padding: 6px 12px;
        border-radius: 16px;
        margin: 4px;
        display: inline-block;
        font-weight: 500;
    }
    .missing-skill {
        background-color: rgba(220, 53, 69, 0.2);
        border: 1px solid #dc3545;
        color: #ef9a9a;
        padding: 6px 12px;
        border-radius: 16px;
        margin: 4px;
        display: inline-block;
        font-weight: 500;
    }
    .priority-high { color: #ef5350; font-weight: bold; }
    .priority-medium { color: #ffc107; font-weight: bold; }
    .priority-low { color: #66bb6a; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

@dataclass
class SkillMatch:
    """Individual skill match details"""
    jd_skill: str
    resume_skill: str
    similarity: float
    category: str
    confidence_level: str
    priority: str = "MEDIUM"
    
    def to_dict(self) -> Dict:
        return {
            'jd_skill': self.jd_skill,
            'resume_skill': self.resume_skill,
            'similarity': self.similarity,
            'category': self.category,
            'confidence_level': self.confidence_level,
            'priority': self.priority
        }

@dataclass
class GapAnalysisResult:
    """Complete gap analysis results"""
    matched_skills: List[SkillMatch]
    partial_matches: List[SkillMatch]
    missing_skills: List[SkillMatch]
    overall_score: float
    category_scores: Dict[str, float]
    similarity_matrix: np.ndarray
    resume_skills: List[str]
    jd_skills: List[str]
    
    def get_statistics(self) -> Dict:
        total = len(self.jd_skills)
        return {
            'total_required_skills': total,
            'matched_count': len(self.matched_skills),
            'partial_count': len(self.partial_matches),
            'missing_count': len(self.missing_skills),
            'match_percentage': (len(self.matched_skills) / total * 100) if total > 0 else 0,
            'overall_score': self.overall_score * 100
        }


class SentenceBERTEncoder:
    """Handles BERT embedding generation using Sentence-BERT"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize Sentence-BERT model
        """
        self.model_name = model_name
        self.logger = self._setup_logger()
        self.embedding_cache = {}
        
        try:
            self.logger.info(f"Loading model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            self.logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dimension}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def encode_skills(self, skills: List[str], use_cache: bool = True, 
                     show_progress: bool = False) -> np.ndarray:
        if not skills:
            raise ValueError("Skills list cannot be empty")
        
        # Check cache
        if use_cache:
            cached_embeddings = []
            uncached_skills = []
            uncached_indices = []
            
            for i, skill in enumerate(skills):
                if skill in self.embedding_cache:
                    cached_embeddings.append(self.embedding_cache[skill])
                else:
                    uncached_skills.append(skill)
                    uncached_indices.append(i)
            
            # Encode uncached skills
            if uncached_skills:
                new_embeddings = self.model.encode(
                    uncached_skills, 
                    show_progress_bar=show_progress,
                    batch_size=32
                )
                
                # Update cache
                for skill, embedding in zip(uncached_skills, new_embeddings):
                    self.embedding_cache[skill] = embedding
                
                # Combine cached and new
                all_embeddings = [None] * len(skills)
                cached_idx = 0
                uncached_idx = 0
                
                for i in range(len(skills)):
                    if i in uncached_indices:
                        all_embeddings[i] = new_embeddings[uncached_idx]
                        uncached_idx += 1
                    else:
                        all_embeddings[i] = cached_embeddings[cached_idx]
                        cached_idx += 1
                
                return np.array(all_embeddings)
            else:
                return np.array(cached_embeddings)
        else:
            # Encode without cache
            embeddings = self.model.encode(
                skills, 
                show_progress_bar=show_progress,
                batch_size=32
            )
            return embeddings
    
    def get_embedding_for_skill(self, skill: str) -> np.ndarray:
        """Get embedding for a single skill"""
        if skill in self.embedding_cache:
            return self.embedding_cache[skill]
        
        embedding = self.model.encode([skill])[0]
        self.embedding_cache[skill] = embedding
        return embedding
    
    def clear_cache(self):
        """Clear embedding cache"""
        self.embedding_cache.clear()
        self.logger.info("Embedding cache cleared")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('BERTEncoder')
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger


class SimilarityCalculator:
    """Compute similarity scores between skills"""
    
    def __init__(self):
        self.logger = self._setup_logger()
    
    def compute_cosine_similarity(self, embedding1: np.ndarray, 
                                  embedding2: np.ndarray) -> float:
        # Reshape if needed
        if embedding1.ndim == 1:
            embedding1 = embedding1.reshape(1, -1)
        if embedding2.ndim == 1:
            embedding2 = embedding2.reshape(1, -1)
        
        similarity = cosine_similarity(embedding1, embedding2)[0][0]
        return float(similarity)
    
    def compute_similarity_matrix(self, resume_embeddings: np.ndarray,
                                  jd_embeddings: np.ndarray) -> np.ndarray:
        self.logger.info(f"Computing similarity matrix: {resume_embeddings.shape} x {jd_embeddings.shape}")
        if resume_embeddings.shape[0] == 0 or jd_embeddings.shape[0] == 0:
            return np.array([[]]) # Return empty matrix if no skills
        similarity_matrix = cosine_similarity(resume_embeddings, jd_embeddings)
        self.logger.info(f"Similarity matrix computed: {similarity_matrix.shape}")
        return similarity_matrix
    
    def find_best_matches(self, similarity_matrix: np.ndarray, 
                         threshold: float = 0.5) -> List[Tuple[int, int, float]]:
        matches = []
        n_resume, n_jd = similarity_matrix.shape
        
        for jd_idx in range(n_jd):
            best_resume_idx = np.argmax(similarity_matrix[:, jd_idx])
            best_similarity = similarity_matrix[best_resume_idx, jd_idx]
            
            if best_similarity >= threshold:
                matches.append((best_resume_idx, jd_idx, best_similarity))
        
        return matches
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('SimilarityCalculator')
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger


class SkillGapAnalyzer:
    """Main skill gap analysis engine"""
    
    def __init__(self, encoder: SentenceBERTEncoder, calculator: SimilarityCalculator,
                 strong_threshold: float = 0.80, partial_threshold: float = 0.50):
        self.encoder = encoder
        self.calculator = calculator
        self.strong_threshold = strong_threshold
        self.partial_threshold = partial_threshold
        self.logger = self._setup_logger()
    
    def analyze(self, resume_skills: List[str], jd_skills: List[str],
               skill_categories: Optional[Dict[str, str]] = None) -> GapAnalysisResult:
        self.logger.info(f"Starting gap analysis: {len(resume_skills)} resume skills vs {len(jd_skills)} JD skills")
        
        # Validate inputs
        if not resume_skills or not jd_skills:
            raise ValueError("Both resume_skills and jd_skills must be non-empty")
        
        # Step 1: Generate embeddings
        self.logger.info("Step 1: Generating BERT embeddings...")
        resume_embeddings = self.encoder.encode_skills(resume_skills, show_progress=True)
        jd_embeddings = self.encoder.encode_skills(jd_skills, show_progress=True)
        
        # Step 2: Compute similarity matrix
        self.logger.info("Step 2: Computing similarity matrix...")
        similarity_matrix = self.calculator.compute_similarity_matrix(
            resume_embeddings, 
            jd_embeddings
        )
        
        # Step 3: Classify matches
        self.logger.info("Step 3: Classifying skill matches...")
        matched_skills = []
        partial_matches = []
        missing_skills = []
        
        for jd_idx, jd_skill in enumerate(jd_skills):
            # Find best matching resume skill
            best_resume_idx = np.argmax(similarity_matrix[:, jd_idx])
            best_similarity = float(similarity_matrix[best_resume_idx, jd_idx])
            resume_skill = resume_skills[best_resume_idx]
            
            # Get category
            category = skill_categories.get(jd_skill, 'other') if skill_categories else 'other'
            
            # Classify based on similarity
            if best_similarity >= self.strong_threshold:
                match = SkillMatch(
                    jd_skill=jd_skill,
                    resume_skill=resume_skill,
                    similarity=best_similarity,
                    category='STRONG_MATCH',
                    confidence_level='HIGH',
                    priority='LOW'
                )
                matched_skills.append(match)
                
            elif best_similarity >= self.partial_threshold:
                match = SkillMatch(
                    jd_skill=jd_skill,
                    resume_skill=resume_skill,
                    similarity=best_similarity,
                    category='PARTIAL_MATCH',
                    confidence_level='MEDIUM',
                    priority='MEDIUM'
                )
                partial_matches.append(match)
                
            else:
                match = SkillMatch(
                    jd_skill=jd_skill,
                    resume_skill=resume_skill,
                    similarity=best_similarity,
                    category='MISSING',
                    confidence_level='LOW',
                    priority='HIGH'
                )
                missing_skills.append(match)
        
        # Step 4: Calculate overall score
        overall_score = self._calculate_overall_score(similarity_matrix)
        
        # Step 5: Calculate category scores
        category_scores = self._calculate_category_scores(
            matched_skills, partial_matches, missing_skills
        )
        
        self.logger.info(f"Analysis complete: {len(matched_skills)} matched, "
                        f"{len(partial_matches)} partial, {len(missing_skills)} missing")
        
        return GapAnalysisResult(
            matched_skills=matched_skills,
            partial_matches=partial_matches,
            missing_skills=missing_skills,
            overall_score=overall_score,
            category_scores=category_scores,
            similarity_matrix=similarity_matrix,
            resume_skills=resume_skills,
            jd_skills=jd_skills
        )
    
    def _calculate_overall_score(self, similarity_matrix: np.ndarray) -> float:
        """Calculate overall match score"""
        # Take maximum similarity for each JD skill
        max_similarities = similarity_matrix.max(axis=0)
        # Average of all maximum similarities
        overall_score = float(np.mean(max_similarities))
        return overall_score
    
    def _calculate_category_scores(self, matched: List[SkillMatch],
                                   partial: List[SkillMatch],
                                   missing: List[SkillMatch]) -> Dict[str, float]:
        """Calculate scores by category"""
        category_scores = {}
        
        all_skills = matched + partial + missing
        categories = set(skill.category for skill in all_skills)
        
        for category in categories:
            cat_skills = [s for s in all_skills if s.category == category]
            if cat_skills:
                avg_similarity = np.mean([s.similarity for s in cat_skills])
                category_scores[category] = float(avg_similarity)
        
        return category_scores
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('SkillGapAnalyzer')
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger


class SkillRanker:
    """Rank skills by importance and priority"""
    
    def __init__(self):
        self.logger = self._setup_logger()
    
    def rank_by_importance(self, skills: List[SkillMatch], 
                          importance_weights: Optional[Dict[str, float]] = None) -> List[SkillMatch]:
        """
        Rank skills by importance
        
        Args:
            skills: List of SkillMatch objects
            importance_weights: Optional weights for different factors
            
        Returns:
            Ranked list of skills
        """
        if not importance_weights:
            importance_weights = {
                'similarity': 0.4,
                'category': 0.3,
                'priority': 0.3
            }
        
        def calculate_importance_score(skill: SkillMatch) -> float:
            # Similarity score
            sim_score = skill.similarity
            
            # Category score (missing skills are more important to address)
            if skill.category == 'MISSING':
                cat_score = 1.0
            elif skill.category == 'PARTIAL_MATCH':
                cat_score = 0.6
            else:
                cat_score = 0.2
            
            # Priority score
            priority_map = {'HIGH': 1.0, 'MEDIUM': 0.6, 'LOW': 0.3}
            pri_score = priority_map.get(skill.priority, 0.5)
            
            # Weighted combination
            importance = (
                importance_weights['similarity'] * sim_score +
                importance_weights['category'] * cat_score +
                importance_weights['priority'] * pri_score
            )
            
            return importance
        
        # Sort by importance (descending)
        ranked_skills = sorted(skills, key=calculate_importance_score, reverse=True)
        
        self.logger.info(f"Ranked {len(skills)} skills by importance")
        return ranked_skills
    
    def categorize_by_urgency(self, missing_skills: List[SkillMatch]) -> Dict[str, List[SkillMatch]]:
        """
        Categorize missing skills by urgency
        
        Returns:
            Dictionary with 'critical', 'important', and 'beneficial' keys
        """
        categorized = {
            'critical': [],
            'important': [],
            'beneficial': []
        }
        
        for skill in missing_skills:
            if skill.priority == 'HIGH' or skill.similarity < 0.3:
                categorized['critical'].append(skill)
            elif skill.priority == 'MEDIUM' or skill.similarity < 0.4:
                categorized['important'].append(skill)
            else:
                categorized['beneficial'].append(skill)
        
        return categorized
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('SkillRanker')
        if not logger.handlers:
            logger.setLevel(logging.INFO)
        return logger


class GapVisualizer:
    """Create visualizations for gap analysis"""
    
    @staticmethod
    def create_similarity_heatmap(similarity_matrix: np.ndarray,
                                     resume_skills: List[str],
                                     jd_skills: List[str]) -> go.Figure:
        """Create a clean, interactive similarity heatmap."""

        # Limit display to avoid overcrowding
        max_display = 20
        display_resume = resume_skills[:max_display]
        display_jd = jd_skills[:max_display]
        display_matrix = similarity_matrix[:max_display, :max_display]

    # Create hover text to show the skill pair and score
        hover_text = []
        for r_skill in display_resume:
            row = []
            for j_skill in display_jd:
                row.append(f"Resume: {r_skill}<br>JD: {j_skill}")
            hover_text.append(row)

        fig = go.Figure(data=go.Heatmap(
            z=display_matrix,
            x=display_jd,
            y=display_resume,
            colorscale='RdYlGn',
            zmid=0.5,
            texttemplate='%{z:.2f}',
            textfont={"size": 10},
            hoverinfo='z+text',
            hovertext=hover_text,
            colorbar=dict(
                title=dict(
                    text="Similarity",
                    side="right"
                )
            )
        ))

        # 3. Refine the layout for better readability
        fig.update_layout(
            title_text=f"Skill Similarity Heatmap (Top {min(max_display, len(resume_skills))} Skills)",
            title_x=0.5,
            xaxis_title="Job Description Skills",
            yaxis_title="Resume Skills",
            height=700,
            width=900,
            xaxis={'side': 'bottom', 'tickangle': -45},
            yaxis={'autorange': 'reversed'},
            margin=dict(l=120, r=40, b=120, t=80) # Adjust margins
        )

        return fig
    
    @staticmethod
    def create_match_distribution_pie(analysis_result: GapAnalysisResult) -> go.Figure:
        """Create pie chart for match distribution"""
        
        stats = analysis_result.get_statistics()
        
        labels = ['Strong Matches', 'Partial Matches', 'Missing Skills']
        values = [stats['matched_count'], stats['partial_count'], stats['missing_count']]
        colors = ['#28a745', '#ffc107', '#dc3545']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors),
            hole=0.3,
            textposition='auto',
            textinfo='label+percent+value'
        )])
        
        fig.update_layout(
            title="Skill Match Distribution",
            height=500,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_skill_comparison_bar(analysis_result: GapAnalysisResult, top_n: int = 15) -> go.Figure:
        """Create bar chart comparing skill similarities"""
        
        all_matches = (analysis_result.matched_skills + 
                      analysis_result.partial_matches + 
                      analysis_result.missing_skills)
        
        # Sort by similarity
        all_matches_sorted = sorted(all_matches, key=lambda x: x.similarity, reverse=True)[:top_n]
        
        skills = [m.jd_skill for m in all_matches_sorted]
        similarities = [m.similarity * 100 for m in all_matches_sorted]
        colors_map = {'STRONG_MATCH': '#28a745', 'PARTIAL_MATCH': '#ffc107', 'MISSING': '#dc3545'}
        colors = [colors_map[m.category] for m in all_matches_sorted]
        
        fig = go.Figure(data=[go.Bar(
            y=skills,
            x=similarities,
            orientation='h',
            marker=dict(color=colors),
            text=[f"{s:.1f}%" for s in similarities],
            textposition='auto'
        )])
        
        fig.update_layout(
            title=f"Top {top_n} Skills by Similarity Score",
            xaxis_title="Similarity Score (%)",
            yaxis_title="Skills",
            height=600,
            yaxis=dict(autorange="reversed"),
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_gap_priority_chart(missing_skills: List[SkillMatch]) -> go.Figure:
        """Create chart showing gap priorities"""
        
        if not missing_skills:
            # Return empty figure if no missing skills
            fig = go.Figure()
            fig.update_layout(
                title="No Missing Skills",
                annotations=[dict(
                    text="All required skills are matched!",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=20)
                )]
            )
            return fig
        
        # Sort by similarity (lower similarity = higher priority)
        sorted_skills = sorted(missing_skills, key=lambda x: x.similarity)[:15]
        
        skills = [s.jd_skill for s in sorted_skills]
        similarities = [s.similarity * 100 for s in sorted_skills]
        priorities = [s.priority for s in sorted_skills]
        
        priority_colors = {
            'HIGH': '#dc3545',
            'MEDIUM': '#ffc107',
            'LOW': '#28a745'
        }
        colors = [priority_colors[p] for p in priorities]
        
        fig = go.Figure(data=[go.Bar(
            y=skills,
            x=similarities,
            orientation='h',
            marker=dict(color=colors),
            text=[f"{s:.1f}% - {p}" for s, p in zip(similarities, priorities)],
            textposition='auto'
        )])
        
        fig.update_layout(
            title="Missing Skills by Priority",
            xaxis_title="Current Similarity (%)",
            yaxis_title="Skills",
            height=500,
            yaxis=dict(autorange="reversed")
        )
        
        return fig
    
    @staticmethod
    def create_overall_score_gauge(overall_score: float) -> go.Figure:
        """Create gauge chart for overall match score"""
        
        score_percentage = overall_score * 100
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score_percentage,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Overall Match Score", 'font': {'size': 24}},
            delta={'reference': 70, 'increasing': {'color': "green"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': '#ffcccc'},
                    {'range': [40, 70], 'color': '#ffffcc'},
                    {'range': [70, 100], 'color': '#ccffcc'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        return fig


class ReportGenerator:
    """Generate comprehensive reports"""
    
    def __init__(self):
        self.timestamp = datetime.now()
    
    def generate_text_report(self, analysis_result: GapAnalysisResult) -> str:
        """Generate detailed text report"""
        
        stats = analysis_result.get_statistics()
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("SKILL GAP ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"\nGenerated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("-" * 80)
        report_lines.append("EXECUTIVE SUMMARY")
        report_lines.append("-" * 80)
        report_lines.append(f"Overall Match Score: {stats['overall_score']:.1f}%")
        report_lines.append(f"Total Required Skills: {stats['total_required_skills']}")
        report_lines.append(f"Matched Skills: {stats['matched_count']} ({stats['match_percentage']:.1f}%)")
        report_lines.append(f"Partial Matches: {stats['partial_count']}")
        report_lines.append(f"Missing Skills: {stats['missing_count']}")
        report_lines.append("")
        
        # Strong Matches
        if analysis_result.matched_skills:
            report_lines.append("-" * 80)
            report_lines.append("✓ STRONG MATCHES (Similarity ≥ 80%)")
            report_lines.append("-" * 80)
            for match in analysis_result.matched_skills:
                report_lines.append(f"  • {match.jd_skill}")
                report_lines.append(f"    Resume: {match.resume_skill}")
                report_lines.append(f"    Similarity: {match.similarity*100:.1f}%")
                report_lines.append("")
        
        # Partial Matches
        if analysis_result.partial_matches:
            report_lines.append("-" * 80)
            report_lines.append("⚠ PARTIAL MATCHES (Similarity 50-80%)")
            report_lines.append("-" * 80)
            for match in analysis_result.partial_matches:
                report_lines.append(f"  • {match.jd_skill}")
                report_lines.append(f"    Closest: {match.resume_skill}")
                report_lines.append(f"    Similarity: {match.similarity*100:.1f}%")
                report_lines.append(f"    Recommendation: Strengthen knowledge in {match.jd_skill}")
                report_lines.append("")
        
        # Missing Skills
        if analysis_result.missing_skills:
            report_lines.append("-" * 80)
            report_lines.append("✗ CRITICAL GAPS (Similarity < 50%)")
            report_lines.append("-" * 80)
            for match in analysis_result.missing_skills:
                report_lines.append(f"  • {match.jd_skill} - {match.priority} PRIORITY")
                report_lines.append(f"    Current closest: {match.resume_skill} ({match.similarity*100:.1f}%)")
                report_lines.append(f"    Action: Acquire {match.jd_skill} through training/certification")
                report_lines.append("")
        
        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def generate_csv_report(self, analysis_result: GapAnalysisResult) -> str:
        """Generate CSV report"""
        
        data = []
        
        # Add all matches
        for match in analysis_result.matched_skills:
            data.append({
                'JD Skill': match.jd_skill,
                'Resume Skill': match.resume_skill,
                'Similarity (%)': f"{match.similarity*100:.2f}",
                'Category': match.category,
                'Priority': match.priority,
                'Status': 'Matched'
            })
        
        for match in analysis_result.partial_matches:
            data.append({
                'JD Skill': match.jd_skill,
                'Resume Skill': match.resume_skill,
                'Similarity (%)': f"{match.similarity*100:.2f}",
                'Category': match.category,
                'Priority': match.priority,
                'Status': 'Partial'
            })
        
        for match in analysis_result.missing_skills:
            data.append({
                'JD Skill': match.jd_skill,
                'Resume Skill': match.resume_skill,
                'Similarity (%)': f"{match.similarity*100:.2f}",
                'Category': match.category,
                'Priority': match.priority,
                'Status': 'Missing'
            })
        
        df = pd.DataFrame(data)
        return df.to_csv(index=False)
    
    def generate_json_report(self, analysis_result: GapAnalysisResult) -> str:
        """Generate JSON report"""
        
        stats = analysis_result.get_statistics()
        
        report_data = {
            'timestamp': self.timestamp.isoformat(),
            'statistics': stats,
            'matched_skills': [match.to_dict() for match in analysis_result.matched_skills],
            'partial_matches': [match.to_dict() for match in analysis_result.partial_matches],
            'missing_skills': [match.to_dict() for match in analysis_result.missing_skills],
            'category_scores': analysis_result.category_scores,
            'resume_skills': analysis_result.resume_skills,
            'jd_skills': analysis_result.jd_skills
        }
        
        return json.dumps(report_data, indent=2)


class LearningPathGenerator:
    """Generate personalized learning paths for skill gaps"""
    
    def __init__(self):
        self.resource_database = self._initialize_resources()
    
    def _initialize_resources(self) -> Dict:
        """Initialize learning resources database"""
        return {
            'Python': {
                'difficulty': 'Medium',
                'time_estimate': '4-8 weeks',
                'resources': [
                    'Python for Everybody (Coursera)',
                    'Automate the Boring Stuff with Python',
                    'Official Python Tutorial'
                ]
            },
            'Machine Learning': {
                'difficulty': 'Hard',
                'time_estimate': '12-16 weeks',
                'prerequisites': ['Python', 'Statistics'],
                'resources': [
                    'Andrew Ng Machine Learning Course',
                    'Hands-on Machine Learning with Scikit-Learn',
                    'Fast.ai Practical Deep Learning'
                ]
            },
            'TensorFlow': {
                'difficulty': 'Medium',
                'time_estimate': '6-8 weeks',
                'prerequisites': ['Python', 'Machine Learning'],
                'resources': [
                    'TensorFlow Developer Certificate',
                    'Deep Learning Specialization',
                    'TensorFlow Official Tutorials'
                ]
            },
            'AWS': {
                'difficulty': 'Medium',
                'time_estimate': '8-12 weeks',
                'resources': [
                    'AWS Cloud Practitioner Certification',
                    'AWS Solutions Architect Associate',
                    'AWS Free Tier Hands-on Labs'
                ]
            },
            'Docker': {
                'difficulty': 'Medium',
                'time_estimate': '2-4 weeks',
                'resources': [
                    'Docker Official Documentation',
                    'Docker Mastery Course',
                    'Docker for Developers'
                ]
            }
        }
    
    def generate_path(self, missing_skills: List[SkillMatch],
                     current_skills: List[str]) -> List[Dict]:
        """Generate learning path for missing skills"""
        
        learning_plan = []
        
        # Sort by priority
        sorted_skills = sorted(missing_skills, 
                              key=lambda x: (0 if x.priority == 'HIGH' else 1 if x.priority == 'MEDIUM' else 2,
                                           x.similarity))
        
        for skill_match in sorted_skills:
            skill = skill_match.jd_skill
            
            plan_item = {
                'skill': skill,
                'current_similarity': skill_match.similarity,
                'priority': skill_match.priority,
                'difficulty': 'Unknown',
                'time_estimate': 'Varies',
                'resources': [],
                'prerequisites': [],
                'missing_prerequisites': []
            }
            
            # Check if we have info for this skill
            if skill in self.resource_database:
                resource_info = self.resource_database[skill]
                plan_item['difficulty'] = resource_info.get('difficulty', 'Unknown')
                plan_item['time_estimate'] = resource_info.get('time_estimate', 'Varies')
                plan_item['resources'] = resource_info.get('resources', [])
                plan_item['prerequisites'] = resource_info.get('prerequisites', [])
                
                # Check prerequisites
                missing_prereqs = []
                for prereq in plan_item['prerequisites']:
                    if prereq.lower() not in [s.lower() for s in current_skills]:
                        missing_prereqs.append(prereq)
                
                plan_item['missing_prerequisites'] = missing_prereqs
            else:
                plan_item['resources'] = [f'Search for "{skill}" courses online']
            
            learning_plan.append(plan_item)
        
        return learning_plan


class CompleteSkillGapApp:
    """Complete Milestone 3 Streamlit Application"""
    
    def __init__(self):
        # Initialize components
        self.encoder = SentenceBERTEncoder()
        self.calculator = SimilarityCalculator()
        self.visualizer = GapVisualizer()
        self.report_generator = ReportGenerator()
        self.learning_path_gen = LearningPathGenerator()
        
        # Initialize session state
        if 'analysis_result' not in st.session_state:
            st.session_state.analysis_result = None
        if 'resume_skills' not in st.session_state:
            st.session_state.resume_skills = []
        if 'jd_skills' not in st.session_state:
            st.session_state.jd_skills = []
    
    def run(self):
        """Run the complete application"""
        
        st.title("🎯 AI Skill Gap Analyzer - Milestone 3")
        st.markdown("### Advanced Skill Gap Analysis with BERT-based Semantic Matching")
        
        # Main tabs
        tabs = st.tabs([
            "🔍 Gap Analysis",
            "📊 Visualizations",
            "📈 Similarity Matrix",
            "🎓 Learning Path",
            "📥 Export Reports",
            "⚙️ Settings"
        ])
        
        with tabs[0]:
            self._gap_analysis_tab()
        
        with tabs[1]:
            self._visualizations_tab()
        
        with tabs[2]:
            self._similarity_matrix_tab()
        
        with tabs[3]:
            self._learning_path_tab()
        
        with tabs[4]:
            self._export_tab()
        
        with tabs[5]:
            self._settings_tab()
    
    def _gap_analysis_tab(self):
        """Main gap analysis interface"""
        
        st.header("Skill Gap Analysis")
        
        st.info(
            """
            **How It Works:**
            1.  **Input Skills:** Enter skills from a resume and a job description manually, or upload the JSON files from Milestone 2.
            2.  **Semantic Analysis:** The system generates advanced BERT embeddings to understand the *meaning* behind each skill.
            3.  **Similarity Scoring:** It computes the cosine similarity between every skill pair to find the best matches.
            4.  **Gap Identification:** The app categorizes skills as **Strong Matches**, **Partial Matches**, or **Missing Skills** based on similarity scores.
            5.  **Actionable Plan:** Finally, it provides a personalized learning path to help you close the identified skill gaps.
            """
        )
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Manual Entry", "Upload from Milestone 2"],
            horizontal=True
        )
        
        resume_skills = []
        jd_skills = []
        
        if input_method == "Manual Entry":
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📄 Resume Skills")
                resume_text = st.text_area(
                    "Enter skills (one per line):",
                    height=300,
                    placeholder="Python\nMachine Learning\nSQL\nData Analysis",
                    key="resume_input"
                )
                if resume_text:
                    resume_skills = [s.strip() for s in resume_text.split('\n') if s.strip()]
                    st.info(f"**{len(resume_skills)} skills entered**")
            
            with col2:
                st.subheader("💼 Job Description Skills")
                jd_text = st.text_area(
                    "Enter required skills (one per line):",
                    height=300,
                    placeholder="Python\nDeep Learning\nTensorFlow\nSQL\nAWS",
                    key="jd_input"
                )
                if jd_text:
                    jd_skills = [s.strip() for s in jd_text.split('\n') if s.strip()]
                    st.info(f"**{len(jd_skills)} skills entered**")
        
        elif input_method == "Upload from Milestone 2":
            st.info("Upload JSON file exported from Milestone 2")
            
            col1, col2 = st.columns(2)
            
            with col1:
                resume_file = st.file_uploader("Upload Resume Skills (JSON)", type=['json'], key='resume_json')
                if resume_file:
                    try:
                        data = json.load(resume_file)
                        resume_skills = data.get('skills', {}).get('all_skills', [])
                        st.success(f"✅ Loaded {len(resume_skills)} resume skills")
                    except Exception as e:
                        st.error(f"Error loading file: {e}")
            
            with col2:
                jd_file = st.file_uploader("Upload JD Skills (JSON)", type=['json'], key='jd_json')
                if jd_file:
                    try:
                        data = json.load(jd_file)
                        jd_skills = data.get('skills', {}).get('all_skills', [])
                        st.success(f"✅ Loaded {len(jd_skills)} JD skills")
                    except Exception as e:
                        st.error(f"Error loading file: {e}")
        
        # Analysis button
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Analyze Skill Gaps", type="primary", use_container_width=True):
                if not resume_skills or not jd_skills:
                    st.error("⚠️ Please provide both resume and JD skills")
                else:
                    self._perform_analysis(resume_skills, jd_skills)
        
        # Display results if available
        if st.session_state.analysis_result:
            st.markdown("---")
            self._display_analysis_results(st.session_state.analysis_result)
    
    def _perform_analysis(self, resume_skills: List[str], jd_skills: List[str]):
        """Perform the gap analysis"""
        
        with st.spinner("🔄 Analyzing skills..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Initialize analyzer
                status_text.text("Initializing analyzer...")
                progress_bar.progress(20)
                
                strong_threshold = st.session_state.get('strong_threshold', 0.80)
                partial_threshold = st.session_state.get('partial_threshold', 0.50)
                
                analyzer = SkillGapAnalyzer(
                    self.encoder,
                    self.calculator,
                    strong_threshold=strong_threshold,
                    partial_threshold=partial_threshold
                )
                
                # Step 2: Run analysis
                status_text.text("Running gap analysis...")
                progress_bar.progress(40)
                
                result = analyzer.analyze(resume_skills, jd_skills)
                
                # Step 3: Store results
                progress_bar.progress(80)
                status_text.text("Storing results...")
                
                st.session_state.analysis_result = result
                st.session_state.resume_skills = resume_skills
                st.session_state.jd_skills = jd_skills
                
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                
                st.success("✅ Gap analysis completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.exception(e)
            finally:
                progress_bar.empty()
                status_text.empty()
    
    def _display_analysis_results(self, result: GapAnalysisResult):
        """Display analysis results"""
        
        st.header("📊 Analysis Results")
        
        stats = result.get_statistics()
        
        # Metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Overall Score", f"{stats['overall_score']:.1f}%")
        with col2:
            st.metric("Total Required", stats['total_required_skills'])
        with col3:
            st.metric("✅ Matched", stats['matched_count'])
        with col4:
            st.metric("⚠️ Partial", stats['partial_count'])
        with col5:
            st.metric("❌ Missing", stats['missing_count'])
        
        # Detailed results
        st.markdown("---")
        
        # Strong matches
        if result.matched_skills:
            with st.expander(f"✅ **Strong Matches ({len(result.matched_skills)})**", expanded=True):
                for match in result.matched_skills:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(
                            f'<div class="strong-match">**{match.jd_skill}** ↔ {match.resume_skill}</div>',
                            unsafe_allow_html=True
                        )
                    with col2:
                        st.metric("Similarity", f"{match.similarity*100:.1f}%")
        
        # Partial matches
        if result.partial_matches:
            with st.expander(f"⚠️ **Partial Matches ({len(result.partial_matches)})**"):
                for match in result.partial_matches:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(
                            f'<div class="partial-match">**{match.jd_skill}** ↔ {match.resume_skill}</div>',
                            unsafe_allow_html=True
                        )
                        st.caption(f"💡 Recommendation: Strengthen knowledge in {match.jd_skill}")
                    with col2:
                        st.metric("Similarity", f"{match.similarity*100:.1f}%")
        
        # Missing skills
        if result.missing_skills:
            with st.expander(f"❌ **Missing Skills ({len(result.missing_skills)})**"):
                ranker = SkillRanker()
                categorized = ranker.categorize_by_urgency(result.missing_skills)
                
                if categorized['critical']:
                    st.markdown("**🔴 CRITICAL (High Priority)**")
                    for match in categorized['critical']:
                        st.markdown(
                            f'<div class="missing-skill">**{match.jd_skill}** - Learn this skill!</div>',
                            unsafe_allow_html=True
                        )
                        st.caption(f"Current closest: {match.resume_skill} ({match.similarity*100:.1f}%)")
                
                if categorized['important']:
                    st.markdown("**🟡 IMPORTANT (Medium Priority)**")
                    for match in categorized['important']:
                        st.markdown(
                            f'<div class="missing-skill">**{match.jd_skill}**</div>',
                            unsafe_allow_html=True
                        )
                
                if categorized['beneficial']:
                    st.markdown("**🟢 BENEFICIAL (Nice to Have)**")
                    for match in categorized['beneficial']:
                        st.markdown(f"• {match.jd_skill}")
    
    def _visualizations_tab(self):
        """Visualizations tab"""
        
        if not st.session_state.analysis_result:
            st.info("👈 Please run gap analysis first in the 'Gap Analysis' tab")
            return
        
        result = st.session_state.analysis_result
        
        st.header("📊 Visual Analytics")
        
        # Overall score gauge
        st.subheader("Overall Match Score")
        fig_gauge = self.visualizer.create_overall_score_gauge(result.overall_score)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution pie chart
            st.subheader("Match Distribution")
            fig_pie = self.visualizer.create_match_distribution_pie(result)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Skill comparison bar
            st.subheader("Top Skills Comparison")
            top_n = st.slider("Number of skills to display:", 5, 30, 15, key='bar_top_n')
            fig_bar = self.visualizer.create_skill_comparison_bar(result, top_n)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Gap priority chart
        if result.missing_skills:
            st.subheader("Missing Skills Priority Analysis")
            fig_priority = self.visualizer.create_gap_priority_chart(result.missing_skills)
            st.plotly_chart(fig_priority, use_container_width=True)
    
    def _similarity_matrix_tab(self):
        """Similarity matrix tab"""
        
        if not st.session_state.analysis_result:
            st.info("👈 Please run gap analysis first in the 'Gap Analysis' tab")
            return
        
        result = st.session_state.analysis_result
        
        st.header("📈 Similarity Matrix Analysis")
        
        st.markdown("""
        This heatmap shows the semantic similarity between all resume skills and job description skills.
        - **Green**: High similarity (strong match)
        - **Yellow**: Medium similarity (partial match)
        - **Red**: Low similarity (skill gap)
        """)
        
        # Heatmap
        fig_heatmap = self.visualizer.create_similarity_heatmap(
            result.similarity_matrix,
            result.resume_skills,
            result.jd_skills
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Detailed matrix view
        with st.expander("📋 View Detailed Similarity Matrix"):
            # Create DataFrame
            df_matrix = pd.DataFrame(
                result.similarity_matrix,
                index=result.resume_skills,
                columns=result.jd_skills
            )
            
            # Format as percentages
            df_display = df_matrix.applymap(lambda x: f"{x*100:.1f}%")
            
            st.dataframe(df_display, use_container_width=True)
            
            # Download matrix
            csv_matrix = df_matrix.to_csv()
            st.download_button(
                "📥 Download Similarity Matrix (CSV)",
                csv_matrix,
                f"similarity_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
    
    def _learning_path_tab(self):
        """Learning path recommendations tab"""
        
        if not st.session_state.analysis_result:
            st.info("👈 Please run gap analysis first in the 'Gap Analysis' tab")
            return
        
        result = st.session_state.analysis_result
        
        st.header("🎓 Personalized Learning Path")
        
        if not result.missing_skills:
            st.success("🎉 Congratulations! No skill gaps found. You match all required skills!")
            return
        
        st.markdown("""
        Based on the identified skill gaps, here's your personalized learning path:
        """)
        
        # Generate learning path
        learning_plan = self.learning_path_gen.generate_path(
            result.missing_skills,
            result.resume_skills
        )
        
        # Display learning path
        for i, item in enumerate(learning_plan, 1):
            priority_class = f"priority-{item['priority'].lower()}"
            
            with st.expander(
                f"{i}. {item['skill']} - {item['priority']} Priority", 
                expanded=(i <= 3)
            ):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Current Similarity", f"{item['current_similarity']*100:.1f}%")
                with col2:
                    st.metric("Difficulty", item['difficulty'])
                with col3:
                    st.metric("Est. Time", item['time_estimate'])
                
                # Prerequisites
                if item['missing_prerequisites']:
                    st.warning(f"⚠️ **Prerequisites needed:** {', '.join(item['missing_prerequisites'])}")
                    st.caption("Learn these skills first before tackling this one")
                elif item['prerequisites']:
                    st.success(f"✅ **Prerequisites satisfied:** {', '.join(item['prerequisites'])}")
                
                # Resources
                if item['resources']:
                    st.markdown("**📚 Recommended Resources:**")
                    for resource in item['resources']:
                        st.markdown(f"• {resource}")
                
                # Action plan
                st.markdown(f"""
                **📋 Action Plan:**
                1. Review prerequisites and foundational concepts
                2. Complete at least one recommended course
                3. Build a small project to apply the skill
                4. Add the skill to your resume once proficient
                """)
        
        # Timeline visualization
        st.subheader("📅 Learning Timeline")
        
        timeline_data = []
        for i, item in enumerate(learning_plan[:10]):  # Limit to top 10
            timeline_data.append({
                'Skill': item['skill'],
                'Priority': item['priority'],
                'Weeks': self._estimate_weeks(item['time_estimate'])
            })
        
        df_timeline = pd.DataFrame(timeline_data)
        
        fig_timeline = go.Figure(data=[go.Bar(
            x=df_timeline['Weeks'],
            y=df_timeline['Skill'],
            orientation='h',
            marker=dict(
                color=['red' if p == 'HIGH' else 'orange' if p == 'MEDIUM' else 'green' 
                       for p in df_timeline['Priority']]
            ),
            text=df_timeline['Weeks'].apply(lambda x: f"{x} weeks"),
            textposition='auto'
        )])
        
        fig_timeline.update_layout(
            title="Estimated Learning Timeline",
            xaxis_title="Estimated Weeks",
            yaxis_title="Skill",
            height=400,
            yaxis=dict(autorange="reversed")
        )
        
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    def _estimate_weeks(self, time_str: str) -> int:
        """Extract weeks from time estimate string"""
        import re
        match = re.search(r'(\d+)-?(\d+)?', time_str)
        if match:
            # Return average if range, otherwise the single value
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) else start
            return (start + end) // 2
        return 8  # Default
    
    def _export_tab(self):
        """Export reports tab"""
        
        if not st.session_state.analysis_result:
            st.info("👈 Please run gap analysis first in the 'Gap Analysis' tab")
            return
        
        result = st.session_state.analysis_result
        
        st.header("📥 Export Analysis Reports")
        
        st.markdown("""
        Download your skill gap analysis in various formats:
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Text report
            st.subheader("📄 Text Report")
            st.markdown("Comprehensive text report with all details")
            
            text_report = self.report_generator.generate_text_report(result)
            
            st.download_button(
                label="📥 Download TXT",
                data=text_report,
                file_name=f"skill_gap_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            # CSV report
            st.subheader("📊 CSV Report")
            st.markdown("Spreadsheet format for further analysis")
            
            csv_report = self.report_generator.generate_csv_report(result)
            
            st.download_button(
                label="📥 Download CSV",
                data=csv_report,
                file_name=f"skill_gap_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col3:
            # JSON report
            st.subheader("📋 JSON Report")
            st.markdown("Structured data for integration")
            
            json_report = self.report_generator.generate_json_report(result)
            
            st.download_button(
                label="📥 Download JSON",
                data=json_report,
                file_name=f"skill_gap_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        # Preview reports
        st.markdown("---")
        st.subheader("📖 Report Preview")
        
        preview_tab = st.selectbox(
            "Select report to preview:",
            ["Text Report", "CSV Report", "JSON Report"]
        )
        
        if preview_tab == "Text Report":
            st.text_area("Report Preview", text_report, height=400)
        elif preview_tab == "CSV Report":
            df = pd.read_csv(io.StringIO(csv_report))
            st.dataframe(df, use_container_width=True)
        else:
            st.json(json.loads(json_report))
    
    def _settings_tab(self):
        """Settings and configuration tab"""
        
        st.header("⚙️ Settings & Configuration")
        
        # Similarity thresholds
        st.subheader("🎚️ Similarity Thresholds")
        
        col1, col2 = st.columns(2)
        
        with col1:
            strong_threshold = st.slider(
                "Strong Match Threshold",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get('strong_threshold', 0.80),
                step=0.05,
                help="Minimum similarity for a skill to be considered a strong match"
            )
            st.session_state.strong_threshold = strong_threshold
        
        with col2:
            partial_threshold = st.slider(
                "Partial Match Threshold",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get('partial_threshold', 0.50),
                step=0.05,
                help="Minimum similarity for a skill to be considered a partial match"
            )
            st.session_state.partial_threshold = partial_threshold
        
        st.info(f"""
        **Current Configuration:**
        - Strong Match: Similarity ≥ {strong_threshold:.0%}
        - Partial Match: {partial_threshold:.0%} ≤ Similarity < {strong_threshold:.0%}
        - Missing/Gap: Similarity < {partial_threshold:.0%}
        """)
        
        # Model settings
        st.markdown("---")
        st.subheader("🤖 Model Configuration")
        
        st.info(f"""
        **Current Model:** {self.encoder.model_name}
        **Embedding Dimension:** {self.encoder.embedding_dimension}
        **Cache Size:** {len(self.encoder.embedding_cache)} embeddings
        """)
        
        if st.button("🗑️ Clear Embedding Cache"):
            self.encoder.clear_cache()
            st.success("Cache cleared!")
        
        # About
        st.markdown("---")
        st.subheader("ℹ️ About Milestone 3")
        
        # Inside the _settings_tab function
        st.markdown("""
            **Milestone 3: Skill Gap Analysis & Similarity Matching**

            **Features Implemented:**
            - ✅ BERT-based semantic similarity using Sentence-BERT
            - ✅ Cosine similarity computation
            - ✅ Multi-level skill gap identification
            - ✅ Importance-based ranking system
            - ✅ Interactive similarity matrices
            - ✅ Comprehensive visualizations
            - ✅ Multiple export formats
            - ✅ Learning path recommendations

            **Technologies Used:**
            - Sentence-Transformers (BERT)
            - Scikit-learn (Cosine Similarity)
            - Plotly (Interactive Visualizations)
            - Streamlit (Web Interface)

            **Model Details:**
            - Model: all-MiniLM-L6-v2
            - Embedding Size: 384 dimensions
            - Performance: Fast inference, good accuracy
            - Use Case: Semantic text similarity
            """)


def main():
    """Main application entry point"""
    
    try:
        app = CompleteSkillGapApp()
        app.run()
        
        # Sidebar
        with st.sidebar:
            st.header("🎯 Milestone 3")
            st.markdown("**Skill Gap Analysis**")
            
            st.markdown("---")
            st.subheader("📊 Quick Stats")
            
            if st.session_state.analysis_result:
                result = st.session_state.analysis_result
                stats = result.get_statistics()
                
                st.metric("Overall Match", f"{stats['overall_score']:.1f}%")
                st.metric("Skills Analyzed", stats['total_required_skills'])
                
                st.markdown("**Breakdown:**")
                st.success(f"✅ Matched: {stats['matched_count']}")
                st.warning(f"⚠️ Partial: {stats['partial_count']}")
                st.error(f"❌ Missing: {stats['missing_count']}")
            else:
                st.info("No analysis yet. Start in the Gap Analysis tab!")
            
            st.markdown("---")
            st.subheader("🚀 Quick Actions")
            
            if st.button("🔄 Reset Analysis", use_container_width=True):
                st.session_state.analysis_result = None
                st.session_state.resume_skills = []
                st.session_state.jd_skills = []
                st.rerun()
            
            st.markdown("---")
            st.caption("Milestone 3 - Complete Implementation")
            st.caption("Version 1.0.0")
    
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()