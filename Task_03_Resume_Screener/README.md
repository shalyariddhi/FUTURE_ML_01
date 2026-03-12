# Task 3: Resume / Candidate Screening System

## 📝 Problem Statement
Manual resume screening is time-consuming for large-scale recruitment. This project builds an automated ML system to parse resumes, extract key skills, and rank candidates based on their relevance to a provided job description.

## 🛠️ Tech Stack & Tools
* **Language:** Python 3.11.14
* **NLP Libraries:** NLTK, spaCy (for text cleaning and tokenization)
* **ML Framework:** Scikit-learn (for Vectorization and Similarity)
* **Web Framework:** Streamlit (for the UI)
* **File Processing:** PyPDF2 / pdfminer (to extract text from PDF resumes)

## ⚙️ How It Works
1. **Text Preprocessing:** Resumes are parsed, converted to lowercase, and stripped of punctuation/stopwords.
2. **Feature Extraction:** I used **TF-IDF Vectorization** to convert textual data into numerical vectors.
3. **Similarity Scoring:** The system calculates the **Cosine Similarity** between the Job Description vector and the Candidate Resume vector.
4. **Ranking:** Candidates are ranked from 0% to 100% based on their match score.

## 📊 Key Features
* **Automated Ranking:** Instantly identifies top candidates.
* **Skill Identification:** Highlights matching vs. missing skills.
* **Batch Processing:** Ability to upload multiple resumes simultaneously.

## 📂 Folder Structure
* `app.py`: Main Streamlit application.
* `runtime.txt`: Version of Python used.
* `requirements.txt`: Dependencies for the project.
