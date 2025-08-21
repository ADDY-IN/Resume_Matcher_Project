import os
import io
import re
import tempfile
from pathlib import Path

import streamlit as st
import pdfplumber
from PIL import Image
import pytesseract

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

import spacy
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz
import pandas as pd

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="Resume Matcher", page_icon="🧠", layout="wide")

# -----------------------------
# NLTK ensure
# -----------------------------
def _ensure_nltk():
    try: nltk.data.find("tokenizers/punkt")
    except LookupError: nltk.download("punkt", quiet=True)
    try: nltk.data.find("corpora/stopwords")
    except LookupError: nltk.download("stopwords", quiet=True)
_ensure_nltk()

STOP_WORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()

# -----------------------------
# Cache heavy objects
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_spacy():
    return spacy.load("en_core_web_sm")
nlp = load_spacy()

@st.cache_resource(show_spinner=True)
def load_model():
    return SentenceTransformer("BAAI/bge-base-en")

model = load_model()

# -----------------------------
# Helpers
# -----------------------------
def extract_pdf_text_with_ocr(file_path_or_bytes, enable_ocr=True) -> str:
    """
    1) Try pdfplumber text
    2) If empty & OCR enabled -> rasterize each page and pytesseract
    accepts path (str) or file-like bytes
    """
    text = ""

    # pdfplumber open
    try:
        if isinstance(file_path_or_bytes, (str, Path)):
            with pdfplumber.open(file_path_or_bytes) as pdf:
                page_texts = [(p.extract_text() or "") for p in pdf.pages]
        else:
            with pdfplumber.open(file_path_or_bytes) as pdf:
                page_texts = [(p.extract_text() or "") for p in pdf.pages]
        text = "\n".join(page_texts).strip()
    except Exception:
        text = ""

    if text or not enable_ocr:
        return text

    # OCR fallback
    ocr_texts = []
    try:
        if isinstance(file_path_or_bytes, (str, Path)):
            with pdfplumber.open(file_path_or_bytes) as pdf:
                for page in pdf.pages:
                    pil_img = page.to_image(resolution=300).original
                    pil_img = Image.fromarray(pil_img)
                    ocr_texts.append(pytesseract.image_to_string(pil_img))
        else:
            with pdfplumber.open(file_path_or_bytes) as pdf:
                for page in pdf.pages:
                    pil_img = page.to_image(resolution=300).original
                    pil_img = Image.fromarray(pil_img)
                    ocr_texts.append(pytesseract.image_to_string(pil_img))
        text = "\n".join(ocr_texts).strip()
    except Exception:
        text = text or ""
    return text

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    doc = nlp(text)
    return " ".join([t.text for t in doc if not t.is_stop])

def extract_keywords_from_jd(jd_text: str):
    jd_text_norm = re.sub(r'[^a-zA-Z0-9\n]', " ", jd_text)
    words = word_tokenize(jd_text_norm.lower())
    filtered = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    # preserve first occurrence order
    unique_keywords = list(dict.fromkeys(filtered))

    lines = jd_text.lower().split("\n")
    jd_title = ""
    jd_experience_lines = []
    for line in lines:
        if any(k in line for k in ["title", "jobtitle", "position", "role"]):
            jd_title = line.strip()
        if "experience" in line:
            jd_experience_lines.append(line.strip())
    return jd_title, jd_experience_lines, unique_keywords

def preprocess_text(t: str):
    t = re.sub(r'[^a-zA-Z0-9]', " ", t)
    return word_tokenize(t.lower())

def compute_scores(jd_text: str, resumes: dict,
                   weight_semantics: float = 0.6,
                   weight_keywords: float = 0.4):
    # Heuristic: if JD has lots of skills, bump keyword weight
    _, _, jd_skills = extract_keywords_from_jd(jd_text)
    if len(jd_skills) >= 25:
        weight_semantics, weight_keywords = 0.4, 0.6

    jd_emb = model.encode(jd_text, convert_to_tensor=True, normalize_embeddings=True)
    jd_skills_stem = [STEMMER.stem(w.lower()) for w in jd_skills]

    results = []
    details = {}

    for fname, raw_txt in resumes.items():
        # Cleaned (for embedding)
        r_emb = model.encode(raw_txt, convert_to_tensor=True, normalize_embeddings=True)
        semantic_score = util.cos_sim(jd_emb, r_emb).item() * 100

        # Keyword coverage
        resume_words = preprocess_text(raw_txt)
        resume_stem = [STEMMER.stem(w) for w in resume_words]
        resume_lower = raw_txt.lower()

        matched_keywords = 0
        matched_list, missing_list = [], []
        for kw, kw_st in zip(jd_skills, jd_skills_stem):
            matched_flag = False
            if kw.lower() in resume_lower:
                matched_flag = True
            elif kw_st in resume_stem:
                matched_flag = True
            else:
                for w in resume_words:
                    if fuzz.partial_ratio(kw.lower(), w) > 90:
                        matched_flag = True
                        break
            if matched_flag:
                matched_keywords += 1
                matched_list.append(kw)
            else:
                missing_list.append(kw)

        keyword_score = (matched_keywords / len(jd_skills)) * 100 if jd_skills else 0.0
        final_score = (weight_semantics * semantic_score) + (weight_keywords * keyword_score)

        results.append({
            "Filename": fname,
            "Final Score": round(final_score, 2),
            "Semantic Score": round(semantic_score, 2),
            "Keyword Score": round(keyword_score, 2),
            "Matched Skills": len(matched_list),
            "Missing Skills": len(missing_list),
        })

        details[fname] = {
            "matched": matched_list,
            "missing": missing_list
        }

    results = sorted(results, key=lambda x: x["Final Score"], reverse=True)
    return results, details

def bytes_to_tempfile(uploaded_file) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return tmp.name

# -----------------------------
# UI
# -----------------------------
st.title("🧠 Resume Matcher – HR Assistant")
st.caption("Upload JD (PDF) and multiple resumes (PDF). Get ranked matches with explainability.")

with st.sidebar:
    st.subheader("Settings")
    ocr_enabled = st.toggle("Enable OCR fallback (for scanned PDFs)", value=True)
    w_sem = st.slider("Weight: Semantics", 0.0, 1.0, 0.6, 0.05)
    w_kw = 1.0 - w_sem
    st.write(f"Weight: Keywords = **{w_kw:.2f}**")
    strong_thr = st.slider("Strong Match Threshold (%)", 0.0, 100.0, 85.0, 1.0)

# Uploaders
col1, col2 = st.columns(2)
with col1:
    jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"], accept_multiple_files=False)
with col2:
    resumes_files = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)

process_btn = st.button("⚡ Process")

if process_btn:
    if not jd_file or not resumes_files:
        st.warning("Please upload both JD and at least one resume.")
    else:
        # Extract JD text
        with st.spinner("Reading JD..."):
            jd_path_tmp = bytes_to_tempfile(jd_file)
            jd_text_raw = extract_pdf_text_with_ocr(jd_path_tmp, enable_ocr=ocr_enabled)
            jd_text_clean = clean_text(jd_text_raw)

        # Extract resumes
        resumes_text = {}
        with st.spinner("Reading resumes..."):
            for f in resumes_files:
                fpath = bytes_to_tempfile(f)
                raw = extract_pdf_text_with_ocr(fpath, enable_ocr=ocr_enabled)
                resumes_text[f.name] = clean_text(raw)

        # Compute
        with st.spinner("Scoring..."):
            results, details = compute_scores(jd_text_clean, resumes_text, weight_semantics=w_sem, weight_keywords=w_kw)

        st.success("Done!")

        # Summary metrics
        strong = sum(1 for r in results if r["Final Score"] >= strong_thr)
        st.metric("Strong Matches (≥ threshold)", strong)

        # Table
        df = pd.DataFrame(results)
        st.subheader("Results")
        st.dataframe(df, use_container_width=True, height=420)

        # Download CSV
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button("⬇️ Download Results CSV", csv_buf.getvalue(), file_name="matching_results.csv", mime="text/csv")

        # Details per candidate
        st.subheader("Per-Candidate Details")
        for row in results:
            fn = row["Filename"]
            with st.expander(f"📄 {fn} — {row['Final Score']}%"):
                c1, c2, c3 = st.columns([1,1,2])
                with c1:
                    st.write(f"**Semantic:** {row['Semantic Score']}%")
                with c2:
                    st.write(f"**Keyword:** {row['Keyword Score']}%")
                with c3:
                    badge = "🟢 Strong" if row["Final Score"] >= strong_thr else ("🟡 Potential" if row["Final Score"] >= 70 else "🔴 Weak")
                    st.write(f"**Verdict:** {badge}")

                matched = details[fn]["matched"]
                missing = details[fn]["missing"]

                st.markdown("**Matched Skills**")
                if matched:
                    st.write(", ".join(sorted(set(matched))))
                else:
                    st.write("_None detected_")

                st.markdown("**Missing Skills**")
                if missing:
                    st.write(", ".join(sorted(set(missing))))
                else:
                    st.write("_None_")

# Footer
st.markdown("---")
st.caption("Tip: Threshold ko tweak karke shortlist fast banao. OCR on rakhna scanned resumes ke liye.")
