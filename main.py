# IMPORTING LIBRARIES
import os
import re
import logging
from pathlib import Path

import pdfplumber
import spacy
nlp = spacy.load("en_core_web_sm")

from sentence_transformers import SentenceTransformer, util  # SentenceTransformer is used for generating embeddings
model = SentenceTransformer('BAAI/bge-base-en')

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from rapidfuzz import fuzz
import pandas as pd


# -----------------------------
# Logging (optional but helpful)
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# -----------------------------
# NLTK: conditional downloads
# -----------------------------
def _ensure_nltk_resources():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

_ensure_nltk_resources()

stemmer = PorterStemmer()  # Initialize the Porter Stemmer
stop_words = set(stopwords.words('english'))


# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_FOLDER = BASE_DIR / "data"
OUTPUT_FOLDER = BASE_DIR / "output"
CLEANED_FOLDER = OUTPUT_FOLDER / "cleaned"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
CLEANED_FOLDER.mkdir(parents=True, exist_ok=True)


# -----------------------------
# EXTRACTING DATA
# -----------------------------
jd_path = DATA_FOLDER / "jd.pdf"
with pdfplumber.open(jd_path) as pdf:
    jd_text = "\n".join([(page.extract_text() or "") for page in pdf.pages])

resume_texts = {}
for file in os.listdir(DATA_FOLDER):
    if file.endswith(".pdf") and file != "jd.pdf":
        path = os.path.join(DATA_FOLDER, file)
        try:
            with pdfplumber.open(path) as pdf:
                resume_texts[file] = "\n".join([(page.extract_text() or "") for page in pdf.pages])
        except Exception as e:
            logging.exception(f"Failed to read {file}: {e}")
            resume_texts[file] = ""

# Save raw txt
with open(os.path.join(OUTPUT_FOLDER, "jd.txt"), "w", encoding="utf-8") as f:
    f.write(jd_text.strip())

for filename, text in resume_texts.items():
    with open(os.path.join(OUTPUT_FOLDER, filename.replace(".pdf", ".txt")), "w", encoding="utf-8") as f:
        f.write(text.strip())


# -----------------------------
# CLEANING DATA
# -----------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    doc = nlp(text)
    return " ".join([token.text for token in doc if not token.is_stop])

# Re-make dirs (safe)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CLEANED_FOLDER, exist_ok=True)

for file in os.listdir(OUTPUT_FOLDER):
    if file.endswith(".txt"):
        with open(os.path.join(OUTPUT_FOLDER, file), "r", encoding="utf-8") as f:
            raw = f.read()
            cleaned = clean_text(raw)
        with open(os.path.join(CLEANED_FOLDER, file), "w", encoding="utf-8") as f:
            f.write(cleaned)


# -----------------------------
# EMBEDDINGS (normalized)
# -----------------------------
input_folder = CLEANED_FOLDER
text = {}
for file in os.listdir(input_folder):
    if file.endswith(".txt"):
        with open(os.path.join(input_folder, file), "r", encoding="utf-8") as f:
            text[file] = f.read()
            print("Loaded:", len(text), "cleaned files")

embeddings = {}
for filename, content in text.items():
    embeddings[filename] = model.encode(
        content,
        convert_to_tensor=True,
        normalize_embeddings=True  # <-- IMPORTANT
    )
    print(f"{filename}= embeddings created")

jd_file = "jd.txt"
jd_embeddings = embeddings[jd_file]

similarity_scores = {}
for file, emb in embeddings.items():
    if file != jd_file:
        # Using util.cos_sim with normalized embeddings
        score = util.cos_sim(jd_embeddings, emb).item()
        similarity_scores[file] = round(score * 100, 2)

sorted_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
print("Resume Matching Scores:")
for file, score in sorted_scores:
    print(f"{file}: {score}% match")


# -----------------------------
# KEYWORD EXTRACTION
# -----------------------------
def extract_keywords_from_jd(jd_text: str):
    jd_text = re.sub(r'[^a-zA-Z0-9\n]', " ", jd_text)
    words = word_tokenize(jd_text.lower())
    filtered = [word for word in words if word not in stop_words and len(word) > 2]
    stemmed = [stemmer.stem(word) for word in filtered]

    # Keep unique by original word (simple, stable)
    unique_keywords = list(dict.fromkeys(filtered))

    lines = jd_text.lower().split("\n")
    jd_title = " "
    jd_experience = []
    for line in lines:
        if "title" in line or "jobtitle" in line or "position" in line or "role" in line:
            jd_title = line.strip()
        if "experience" in line:
            jd_experience.append(line.strip())
    return jd_title, jd_experience, unique_keywords


# -----------------------------
# PREPROCESSING TEXT
# -----------------------------
def preprocess_text(t: str):
    t = re.sub(r'[^a-zA-Z0-9]', " ", t)
    return word_tokenize(t.lower())


# -----------------------------
# MATCHING RESUME WITH JD
# -----------------------------
def match_resume(jd_text: str, resume_text: dict, model: SentenceTransformer,
                 weight_semantics: float = 0.3, weight_keywords: float = 0.7):
    jd_title, jd_experience, jd_skills = extract_keywords_from_jd(jd_text)
    jd_embeddings_local = model.encode(
        jd_text,
        convert_to_tensor=True,
        normalize_embeddings=True  # <-- IMPORTANT
    )
    jd_skills_stemmed = [stemmer.stem(word.lower()) for word in jd_skills]
    results = []

    for filename, t in resume_text.items():
        resume_embeddings = model.encode(
            t,
            convert_to_tensor=True,
            normalize_embeddings=True  # <-- IMPORTANT
        )

        # Semantic score (0..1) * 100
        semantic_score = util.cos_sim(jd_embeddings_local, resume_embeddings).item() * 100

        # Keyword coverage
        resume_words = preprocess_text(t)
        resume_stemmed = [stemmer.stem(word) for word in resume_words]
        resume_text_lower = t.lower()

        matched_keywords = 0
        for kw, stemmed_kw in zip(jd_skills, jd_skills_stemmed):
            if kw.lower() in resume_text_lower:
                matched_keywords += 1
                continue
            if stemmed_kw in resume_stemmed:
                matched_keywords += 1
                continue
            for word in resume_words:
                if fuzz.partial_ratio(kw.lower(), word) > 90:
                    matched_keywords += 1
                    break

        keyword_score = (matched_keywords / len(jd_skills)) * 100 if jd_skills else 0
        final_score = (weight_semantics * semantic_score) + (weight_keywords * keyword_score)

        results.append({
            "filename": filename,
            "semantic_score": round(semantic_score, 2),
            "keyword_score": round(keyword_score, 2),
            "final_score": round(final_score, 2),
        })

    return sorted(results, key=lambda x: x["final_score"], reverse=True)


# -----------------------------
# RUN MATCHING
# -----------------------------
jd_text_clean = text["jd.txt"]
resumes_clean = {k: v for k, v in text.items() if k != "jd.txt"}

results = match_resume(jd_text_clean, resumes_clean, model)
for r in results:
    print(r)

results_path = BASE_DIR / "matching_results.csv"
pd.DataFrame(results).to_csv(results_path, index=False)
print("Results saved to", results_path)

top_matches = [r for r in results if r["final_score"] > 90]
for r in top_matches:
    print(r)
