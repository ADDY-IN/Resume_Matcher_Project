# IMPORTING LIBRARIES
import os 
import pdfplumber
import re
import spacy
nlp = spacy.load("en_core_web_sm")
from sentence_transformers import SentenceTransformer, util # SentenceTransformer is used for generating embeddings
model= SentenceTransformer('BAAI/bge-base-en')
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from rapidfuzz import fuzz
import pandas as pd
stemmer = PorterStemmer() #Initialize the Porter Stemmer
stop_words = set(stopwords.words('english'))
nltk.download('punkt')
nltk.download('stopwords')  


# EXTRACTING DATA


data_folder = "E:\\Resume_Macher_Project\\data"  # TODO: Change the path to your current folder. remove static path
jd_path = os.path.join(data_folder, "jd.pdf")

with pdfplumber.open(jd_path)as pdf:
    jd_text = "\n".join([page.extract_text()for page in pdf.pages])
    
resume_texts = {}
for file in os.listdir(data_folder):
    if file.endswith(".pdf") and file != "jd.pdf":
        path = os.path.join(data_folder, file)
        with pdfplumber.open(path)as pdf:
            resume_texts[file] = "\n".join([page.extract_text() for page in pdf.pages])
            
output_folder = "E:\\Resume_Macher_Project\\output"
os.makedirs(output_folder,exist_ok=True)

with open(os.path.join(output_folder, "jd.txt"), "w", encoding="utf-8")as f:
    f.write(jd_text.strip())
    
for filename,text in resume_texts.items():
    with open(os.path.join(output_folder,filename.replace(".pdf", ".txt")), "w", encoding="utf-8")as f:
        f.write(text.strip())
        
    
# CLEANING DATA

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]"," ",text)
    text = re.sub(r"\s+"," ",text)
    doc = nlp(text)
    return " ".join([token.text for token in doc if not token.is_stop])

input_folder = "E:\\Resume_Macher_Project\\output"
output_folder = "E:\\Resume_Macher_Project\\output\\cleaned"
os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.endswith(".txt"):
        with open(os.path.join(input_folder,file), "r", encoding="utf-8")as f:
            raw= f.read()
            cleaned = clean_text(raw)
        with open(os.path.join(output_folder, file), "w", encoding="utf-8")as f:
            f.write(cleaned)
            
            
# EMBEDDINGS

input_folder = "E:\\Resume_Macher_Project\\output\\cleaned"
text = {}
for file in os.listdir(input_folder):
    if file.endswith(".txt"):
        with open(os.path.join(input_folder, file), "r", encoding="utf-8")as f:
            text[file] = f.read()
            print("Loaded:",len(text),"cleaned files")
            
embeddings = {}
for filename, content in text.items():
    embeddings[filename]= model.encode(content, convert_to_tensor=True)
    print(f"{filename}= embeddings created")
    
jd_file = "jd.txt"
jd_embeddings = embeddings[jd_file]

similarity_scores = {}
for file,emb in embeddings.items():
    if file != jd_file:
        score = util.pytorch_cos_sim(jd_embeddings, emb).item()
        similarity_scores[file] = round(score * 100, 2)
 
sorted_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
print("Resume Matching Scores:")
for file,score in sorted_scores:
    print(f"{file}: {score}%match")
    
# KEYWORD EXTRACTION

def extract_keywords_from_jd(jd_text):
    jd_text = re.sub(r'[^a-zA-Z0-9]', " ", jd_text)
    words = word_tokenize(jd_text.lower())
    filtered = [word for word in words if word not in stop_words and len(word) > 2]
    stemmed = [stemmer.stem(word) for word in filtered]
    unique_keywords = list(set(filtered))
    lines = jd_text.lower().split("\n")
    jd_title = " "
    jd_experience = []
    for line in lines:
        if "title" in line or "jobtitle"in line:
            jd_title = line.strip()
        if "experience" in line:
            jd_experience.append(line.strip())
    return jd_title, jd_experience, unique_keywords


# PREPROCESSING TEXT

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9]', " ", text)
    return word_tokenize(text.lower())

# MATCHING RESUME WITH JD
def match_resume(jd_text, resume_text,model,weight_semantics = 0.3, weight_keywords = 0.7):
    jd_title, jd_experience, jd_skills = extract_keywords_from_jd(jd_text)
    jd_embeddings = model.encode(jd_text, convert_to_tensor=True)
    jd_skils_stemmed = [stemmer.stem(word.lower()) for word in jd_skills]
    results = []
    for filename, text in resume_text.items():
        resume_embeddings = model.encode(text, convert_to_tensor=True)
        semantic_score = util.pytorch_cos_sim(jd_embeddings, resume_embeddings).item()*100
        resume_words = preprocess_text(text)
        resume_stemmed = [stemmer.stem(word)for word in resume_words]
        resume_text_lower = text.lower()
        matched_keywords = 0
        for kw, stemmed_kw in zip(jd_skills, jd_skils_stemmed):
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
            "semantic_score": round(semantic_score,2),
            "keyword_score": round(keyword_score,2),
            "final_score": round(final_score,2),
        })
    return sorted(results, key=lambda x: x["final_score"], reverse=True)

jd_text = text["jd.txt"]
resume = {k:v for k ,v in text.items() if k != "jd.txt"}
results = match_resume(jd_text, resume, model)
for r in results:
    print(r)

df_results = pd.DataFrame(results)
df_results.to_csv("matching_results.csv", index=False)   
print("Results saved to matching_results.csv")
       
top_matches = [r for r in results if r ["final_score"] > 90]
for r in top_matches:
    print(r)
    