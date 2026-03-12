import streamlit as st
import pdfplumber
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.title("AI Resume Screening Tool")

model = SentenceTransformer("all-MiniLM-L6-v2")

SKILLS_DB = [
    "python", "java", "c++", "sql", "mysql", "postgresql",
    "django", "flask", "fastapi", "react", "node.js",
    "aws", "docker", "kubernetes", "git", "linux",
    "machine learning", "tensorflow", "pytorch",
    "data structures", "algorithms"
]

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def clean_text(text):
    return " ".join(text.split())


def calculate_match_score(jd_text, resume_text):
    jd_embedding = model.encode([jd_text])
    resume_embedding = model.encode([resume_text])
    score = cosine_similarity(jd_embedding, resume_embedding)[0][0]
    return float(score)


def extract_skills(text):
    text = text.lower()
    return [skill for skill in SKILLS_DB if skill in text]


def get_fit_label(score):
    """
    Returns a single string for the label and a recommendation separately 
    to make the UI and CSV export cleaner.
    """
    if score >= 0.85:
        return "Excellent Match", "Highly Recommended."
    elif score >= 0.70:
        return "Good Match", "Strong Potential."
    elif score >= 0.50:
        return "Average Match", "Requires Review."
    else:
        return "Low Match", "Not a fit for this role."


jd_text = st.text_area("Paste Job Description")

uploaded_files = st.file_uploader(
    "Upload Resume PDFs",
    accept_multiple_files=True
)

if st.button("Analyze"):
    if not jd_text or not uploaded_files:
        st.error("Please provide both a Job Description and at least one Resume.")
    else:
        results = []
        jd_text_cleaned = clean_text(jd_text)
        jd_skills = extract_skills(jd_text_cleaned)

        for file in uploaded_files:
            resume_text = clean_text(extract_text_from_pdf(file))

            if len(resume_text) < 300:
                st.warning(f"{file.name} has very little readable text.")
                continue

            score = calculate_match_score(jd_text_cleaned, resume_text)
            resume_skills = extract_skills(resume_text)

            matched = [s for s in jd_skills if s in resume_skills]
            missing = [s for s in jd_skills if s not in resume_skills]
            
            # Get label and recommendation
            label, rec = get_fit_label(score)

            results.append({
                "name": file.name,
                "score": score,
                "label": label,
                "recommendation": rec,
                "matched": matched,
                "missing": missing
            })


    results.sort(key=lambda x: x["score"], reverse=True)

        st.subheader("⭐ Top Recommended Candidates")
        for i, c in enumerate(results[:3], start=1):
            st.success(f"{i}. {c['name']} — **{c['label']}** ({c['score']*100:.2f}%)")
    report = pd.DataFrame([
        {
            "Name": r["name"],
            "Match Score (%)": round(r["score"]*100, 2),
            "Fit Level": r["label"],
            "Recommendation": r["recommendation"],
            "Matched Skills": ", ".join(r["matched"]),
            "Missing Skills": ", ".join(r["missing"])
        }
        for r in results
    ])

    st.download_button(
        "⬇️ Download Shortlist Report (CSV)",
        report.to_csv(index=False),
        "resume_shortlist_report.csv",
        "text/csv"
    )
    st.subheader("Detailed Candidate Analysis")
    for r in results:
        with st.expander(f"View Analysis for {r['name']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Match Score", f"{r['score']*100:.1f}%")
                st.write(f"**Fit Status:** {r['label']}")
            with col2:
                st.write(f"**Recommendation:** {r['recommendation']}")

                # Skill Gap Visualization
            st.markdown("---")
            st.write("### 🔍 Skill Gap Analysis")
            if r['matched']:
                st.write(f"✅ **Matched Skills:** {', '.join(r['matched'])}")
                
            if r['missing']:
                # This is the "Pro Tip" logic: alerting the recruiter to missing needs
                st.error(f"❌ **Missing Skills (Gap):** {', '.join(r['missing'])}")
                st.info(f"💡 Suggestion: Candidate needs training or experience in **{r['missing'][0]}**.")
            else:
                 st.balloons()
                st.write("🌟 **Perfect Skill Match!** No gaps identified from the Job Description.")
  
