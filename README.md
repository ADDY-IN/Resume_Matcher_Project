🚀 **Resume Matcher Project**

This is an AI-powered Python application that intelligently matches resumes with job descriptions (JD) using semantic similarity 🤖 and keyword matching 📌. It helps recruiters shortlist the best-fitting candidates in seconds ⚡.

---

✨ **Features**

✅ Semantic & Keyword Matching Combined  
✅ Built with Python, spaCy, and Transformers 🧠  
✅ PDF to Text Conversion using pdfplumber 📄  
✅ Final Match Score in % 🔢  
✅ Outputs Results as CSV 📁  
✅ Easy to Integrate into Any Frontend or Backend 🔗

---

🔍 **How It Works**

1️⃣ Upload JD & Resumes in PDF format  
2️⃣ Extract and Clean Text 📄  
3️⃣ Generate Embeddings using Sentence Transformers 🔤  
4️⃣ Extract Keywords from JD using spaCy 🧠  
5️⃣ Calculate Final Matching Score (Semantic + Keyword) 📊  
6️⃣ Save Results in CSV with Ranked Matches 📈

---

🧰 **Tech Stack**

🐍 Python 3.10+  
📦 sentence-transformers, spaCy, pdfplumber, pandas  
📊 scikit-learn, numpy  
💻 Jupyter Notebook  
☁️ Ready for Flask/FastAPI Integration (Optional)

---

📂 **Project Structure**

Resume_Matcher_Project/
│
├── data/ # Input PDFs (Resumes + JD)
├── output/ # Extracted .txt files
├── processed/ # Cleaned text files
├── notebook.ipynb # Core logic (Jupyter Notebook)
├── matching_results.csv
├── requirements.txt
└── README.md

🚀 **Next Improvements**

🔒 Add resume filtering using keyword rules  
🧠 Enhance model using domain-specific data  
🌐 Add frontend for JD upload and resume viewing  
☁️ Deploy on web with FastAPI + Streamlit

---

👨‍💻 **Contributions Welcome!**  
If you like the project, give it a ⭐ on GitHub.  
Built with ❤️ by **Aditya Kaushik**


