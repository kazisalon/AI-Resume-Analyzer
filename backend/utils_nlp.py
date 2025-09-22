# backend/utils_nlp.py
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import re
import spacy
from typing import List, Dict


nlp = spacy.load('en_core_web_sm') if 'en_core_web_sm' in spacy.util.get_installed_models() else spacy.load('en_core_web_sm')


EMB_MODEL_NAME = 'all-MiniLM-L6-v2'
EMB = SentenceTransformer(EMB_MODEL_NAME)
KW_MODEL = KeyBERT(EMB)


# Basic action verbs for quick improvements
ACTION_VERBS = [
'Led','Managed','Developed','Designed','Implemented','Optimized','Improved','Reduced','Increased',
'Built','Created','Automated','Analyzed','Coordinated','Launched','Delivered','Produced'
]




def get_keywords(text: str, top_n: int = 10) -> List[str]:
    # KeyBERT expects short blocks; we ask for keywords
    try:
        keywords = KW_MODEL.extract_keywords(text, keyphrase_ngram_range=(1,2), stop_words='english', top_n=top_n)
        return [k[0] for k in keywords]
    except Exception:
        # fallback to simple noun chunks
        doc = nlp(text)
        chunks = [chunk.text.strip() for chunk in doc.noun_chunks]
        return chunks[:top_n]




def embed_texts(texts: List[str]):
    return EMB.encode(texts, convert_to_numpy=True)




def text_similarity(a: str, b: str) -> float:
    emb = embed_texts([a, b])
    sim = float(cosine_similarity([emb[0]], [emb[1]])[0][0])
    return sim




def compute_ats_score(resume_text: str, jd_text: str | None) -> Dict:
    """Return a small summary and a score 0-100 based on keyword match and sections present."""
    resume_kw = get_keywords(resume_text, top_n=20)
    jd_kw = get_keywords(jd_text, top_n=20) if jd_text else []

    # Keyword coverage
    matched = 0
    for kw in jd_kw:
        # simple containment check (case-insensitive)
        if re.search(rf'\b{re.escape(kw)}\b', resume_text, flags=re.I):
            matched += 1
    coverage = (matched / len(jd_kw)) if jd_kw else 0.0

    # Section presence heuristic
    sections = ['experience', 'education', 'skills', 'projects', 'certification', 'summary']
    present = {s: bool(re.search(rf'\b{s}\b', resume_text, flags=re.I)) for s in sections}
    section_score = sum(present.values()) / len(sections)

    return {
        "keyword_coverage": round(coverage * 100, 2),
        "section_score": round(section_score * 100, 2),
        "sections_present": present,
        "matched_keywords": matched,
        "total_keywords": len(jd_kw),
    }


def analyze_resume_text(resume_text: str, jd_text: str = None, use_gpt: bool = False) -> Dict:
    """
    Analyze resume text against an optional job description, returning ATS score, skill gaps, and suggestions.
    """
    ats_result = compute_ats_score(resume_text, jd_text)
    resume_keywords = get_keywords(resume_text, top_n=20)
    jd_keywords = get_keywords(jd_text, top_n=20) if jd_text else []

    # Skill gap analysis
    missing_skills = [kw for kw in jd_keywords if not re.search(rf'\b{re.escape(kw)}\b', resume_text, flags=re.I)]
    skill_match_pct = 100 - (len(missing_skills) / len(jd_keywords) * 100) if jd_keywords else 100.0

    # Actionable feedback
    feedback = []
    if missing_skills:
        feedback.append(f"Consider adding these skills or keywords: {', '.join(missing_skills)}.")
    # Suggest more action verbs if few are present
    present_verbs = [verb for verb in ACTION_VERBS if re.search(rf'\b{verb}\b', resume_text, flags=re.I)]
    if len(present_verbs) < 3:
        feedback.append("Try to use more action verbs to strengthen your achievements.")
    # Section feedback
    for section, present in ats_result.get('sections_present', {}).items():
        if not present:
            feedback.append(f"Add a section for {section.title()} to improve completeness.")

    return {
        "ats": ats_result,
        "resume_keywords": resume_keywords,
        "jd_keywords": jd_keywords,
        "missing_skills": missing_skills,
        "skill_match_pct": round(skill_match_pct, 2),
        "feedback": feedback
    }