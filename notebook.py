import os
import pdfplumber

data_folder = "E:\\Resume_Macher_Project\\data"
jd_pdf_path = os.path.join(data_folder, "jd.pdf")
with pdfplumber.open(jd_pdf_path) as pdf:
    text = ""
    for page in pdf.pages:
        text += page.extract_text() + "\n"

resume_text = {}
for file in os.listdir(data_folder):
    if file.endswith(".pdf") and file != "jd.pdf":
        path = os.path.join(data_folder, file)
        resume_text[file] = "" 
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                resume_text[file] += page.extract_text() + "\n"
print("JD length:", len(text))
print("Total resumes extracted:", len(resume_text))

import os
ouput_folder = "E:\\Resume_Macher_Project\\output"
os.makedirs(ouput_folder, exist_ok=True)

with open(os.path.join(ouput_folder, "jd.txt"), "w", encoding="utf-8") as f:
    f.write(text.strip())

for filename, text in resume_text.items():
    with open(os.path.join(ouput_folder, filename.replace(".pdf", ".txt")), "w", encoding="utf-8") as f:
        f.write(text.strip())
print("JD and resumes have been extracted and saved to the output folder.")

input_folder = "E:\\Resume_Macher_Project\\output"
output_folder = "E:\\Resume_Macher_Project\\output\\cleaned"
os.makedirs(output_folder, exist_ok=True)

import re
import spacy
nlp = spacy.load("en_core_web_sm")
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    doc = nlp(text)
    cleaned = " ".join([token.text for token in doc if not token.is_stop])
    return cleaned

for file in os.listdir(input_folder):
    if file.endswith(".txt"):
        with open(os.path.join(input_folder, file), "r", encoding="utf-8") as f:
            raw_text = f.read()
        cleaned = clean_text(raw_text)
        with open(os.path.join(output_folder, file), "w", encoding="utf-8") as f:
            f.write(cleaned)
print("Text cleaning completed. Cleaned files are saved in the cleaned folder.")

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-base-en')
print("Model loaded successfully.")

import os
input_folder = "E:\\Resume_Macher_Project\\output\\cleaned"
text = {}
for file in os.listdir(input_folder):
    if file.endswith(".txt"):
        with open(os.path.join(input_folder, file), "r", encoding="utf-8") as f:
            text[file] = f.read()
print("Loaded:", len(text),"cleaned text files")

embeddings = {}
for filename, content in text.items():
    embeddings[filename] = model.encode(content, convert_to_tensor=True)
    print("embeddings created")

from sentence_transformers import util
jd_file = "jd.txt"
jd_embedding = embeddings[jd_file]
similarity_scores = {}
for file, emb in embeddings.items():
    if file != jd_file:
        score = util.pytorch_cos_sim(jd_embedding, emb).item()
        similarity_scores[file] = round(score * 100, 2)
sorted_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
print("Top Resume Matches:")
for file, score in sorted_scores:
    print(f"{file}:{score}%match")

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re

nltk.download('punkt')
nltk.download('stopwords')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def extract_keywords_from_jd(jd_text):
    jd_text = re.sub(r'[^a-zA-Z0-9 ]', ' ', jd_text)  # Remove special chars
    words = word_tokenize(jd_text.lower())

    filtered = [word for word in words if word not in stop_words and len(word) > 2]
    stemmed = [stemmer.stem(word) for word in filtered]

    unique_keywords = list(set(filtered))  # remove duplicates

    # Naively extract title and experience from text — tweak if needed
    lines = jd_text.lower().split('\n')
    jd_title = ""
    jd_experience = []

    for line in lines:
        if "title" in line or "job title" in line:
            jd_title = line.strip()
        if "experience" in line:
            jd_experience.append(line.strip())

    return jd_title, unique_keywords, jd_experience

from rapidfuzz import fuzz
from sentence_transformers import util
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re

stemmer = PorterStemmer()

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return word_tokenize(text.lower())

def match_resumes(jd_text, resume_texts, model, weight_semantics=0.3, weight_keyword=0.7):
    jd_title, jd_skills, jd_experience = extract_keywords_from_jd(jd_text)

    jd_embedding = model.encode(jd_text, convert_to_tensor=True)
    jd_skills_stemmed = [stemmer.stem(word.lower()) for word in jd_skills]

    results = []

    for filename, text in resume_texts.items():
        resume_embedding = model.encode(text, convert_to_tensor=True)
        semantic_score = util.pytorch_cos_sim(jd_embedding, resume_embedding).item() * 100

        resume_words = preprocess_text(text)
        resume_stemmed = [stemmer.stem(word) for word in resume_words]
        resume_text_lower = text.lower()

        matched_keywords = 0

        for kw, stemmed_kw in zip(jd_skills, jd_skills_stemmed):
            # Phrase match
            if kw.lower() in resume_text_lower:
                matched_keywords += 1
                continue

            # Stem match
            if stemmed_kw in resume_stemmed:
                matched_keywords += 1
                continue

            # Partial fuzzy match fallback
            for word in resume_words:
                if fuzz.partial_ratio(kw.lower(), word) > 90:
                    matched_keywords += 1
                    break

        keyword_score = (matched_keywords / len(jd_skills)) * 100 if jd_skills else 0
        final_score = weight_semantics * semantic_score + weight_keyword * keyword_score

        results.append({
            "filename": filename,
            "semantic_score": round(semantic_score, 2),
            "keyword_score": round(keyword_score, 2),
            "final_score": round(final_score, 2)
        })

    return sorted(results, key=lambda x: x["final_score"], reverse=True)

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

jd_text = text["jd.txt"]
resumes = {k: v for k, v in text.items() if k != "jd.txt"}

results = match_resumes(jd_text, resumes, model)
for r in results:
    print(r)

import pandas as pd
df_results = pd.DataFrame(results)
df_results.to_csv("matching_results.csv", index=False)
print("Results saved to matching_results.csv")

top_matches = [r for r in results if r["final_score"] >= 90]
for r in top_matches:
    print(r)


