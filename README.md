# Resume Matcher Project

A tool to match resumes with a job description using semantic and keyword-based similarity.

## Features

- Extracts text from PDF resumes and JD
- Preprocesses and cleans data
- Generates embeddings using sentence-transformers
- Extracts top keywords from JD
- Combines semantic + keyword scores for matching
- Outputs results to CSV

## How to Run

1. Clone the repo and navigate to folder  
2. Create virtual environment and activate:
python -m venv venv
venv\Scripts\activate
3. Install dependencies:
4. Add all PDFs to `data/` folder  
5. Run `notebook.ipynb` step-by-step  
6. Results will be saved in `matching_results.csv`

## Structure

- `data/` – raw PDFs (JD + resumes)  
- `output/` – extracted text  
- `processed/` – cleaned text  
- `notebook.ipynb` – main logic  
- `requirements.txt` – dependencies  
- `README.md` – project info

## Author

Made by Aditya Kaushik
