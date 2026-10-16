import io
import re
from collections import Counter

import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Career Path Clustering", page_icon="🧭", layout="wide")

CAREER_SKILLS = {
    "Data Science": ["python", "machine learning", "data analysis", "pandas", "numpy", "statistics", "sql", "tensorflow", "power bi", "deep learning", "scikit learn"],
    "Web Development": ["html", "css", "javascript", "react", "angular", "node.js", "flask", "django", "php", "api", "frontend", "backend"],
    "HR / Human Resources": ["recruitment", "communication", "employee relations", "talent acquisition", "payroll", "training", "human resources", "interviewing", "onboarding"],
}


def clean_text(value):
    text = re.sub(r"[^a-zA-Z ]", " ", str(value)).lower()
    return " ".join(word for word in text.split() if word not in ENGLISH_STOP_WORDS and len(word) > 1)


def read_resume(uploaded_file):
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()
    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if name.endswith(".docx"):
        from docx import Document
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    raise ValueError("Please upload a PDF, DOCX, or TXT resume.")


def prepare_dataset(df):
    df = df.copy()
    if "Resume" not in df.columns:
        candidates = [c for c in ["skills", "career_objective", "experience", "education", "description"] if c in df.columns]
        if not candidates:
            raise ValueError("CSV needs a 'Resume' column, or at least one of these columns: skills, career_objective, experience, education, description.")
        df["Resume"] = df[candidates].fillna("").astype(str).agg(" ".join, axis=1)
    df["Resume"] = df["Resume"].fillna("").astype(str)
    df = df[df["Resume"].str.strip().ne("")].reset_index(drop=True)
    if len(df) < 3:
        raise ValueError("Please upload a dataset with at least 3 non-empty resumes.")
    return df


st.title("🧭 Career Path Clustering")
st.markdown("Explore resume patterns with NLP, TF-IDF, and K-Means clustering. You can also test a resume against the dataset.")
st.info("Upload your original **resume.csv** dataset below. The notebook file alone may not contain the dataset, so it must be uploaded separately if it is not saved beside app.py.")

with st.sidebar:
    st.header("1. Training dataset")
    dataset_file = st.file_uploader("Upload resume.csv", type=["csv"], key="dataset")
    st.caption("Expected columns include skills and career_objective, or a single Resume column.")
    k = st.slider("Number of clusters (K)", min_value=2, max_value=10, value=5)

if dataset_file is None:
    st.subheader("How to get started")
    st.write("Use the sidebar to upload `resume.csv`. Once uploaded, this app trains the clustering model and displays the dataset overview, cluster chart, keywords, and resume testing tool.")
    st.write("**Project workflow:** text cleaning → TF-IDF → PCA → K-Means → cluster insights")
    st.stop()

try:
    raw_df = pd.read_csv(dataset_file)
    df = prepare_dataset(raw_df)
    df["cleaned_resume"] = df["Resume"].apply(clean_text)
    df = df[df["cleaned_resume"].str.strip().ne("")].reset_index(drop=True)
    if len(df) < 3:
        st.error("After text cleaning, fewer than 3 usable resumes remain. Check the dataset.")
        st.stop()

    vectorizer = TfidfVectorizer(max_features=2000)
    X = vectorizer.fit_transform(df["cleaned_resume"])
    n_components = min(50, X.shape[0] - 1, X.shape[1] - 1)
    if n_components < 1:
        st.error("Not enough distinct text features to train the model. Add more varied resume text.")
        st.stop()
    X_pca = PCA(n_components=n_components, random_state=42).fit_transform(X.toarray())
    actual_k = min(k, len(df) - 1)
    model = KMeans(n_clusters=actual_k, random_state=42, n_init=10)
    df["Cluster"] = model.fit_predict(X_pca)
    terms = vectorizer.get_feature_names_out()

    c1, c2, c3 = st.columns(3)
    c1.metric("Usable resumes", f"{len(df):,}")
    c2.metric("Clusters", actual_k)
    c3.metric("Text features", f"{X.shape[1]:,}")

    tab1, tab2, tab3 = st.tabs(["📊 Dataset & charts", "🔎 Cluster insights", "📄 Test a resume"])
    with tab1:
        st.subheader("Dataset preview")
        st.dataframe(df[["Resume", "Cluster"]].head(20), use_container_width=True)
        counts = df["Cluster"].value_counts().sort_index().rename_axis("Cluster").reset_index(name="Resume count")
        left, right = st.columns(2)
        with left:
            st.plotly_chart(px.bar(counts, x="Cluster", y="Resume count", title="Resumes in each cluster"), use_container_width=True)
        with right:
            lengths = df["Resume"].str.split().str.len().rename("Word count").to_frame()
            st.plotly_chart(px.histogram(lengths, x="Word count", nbins=30, title="Resume length distribution"), use_container_width=True)
        if 2 <= df["Cluster"].nunique() < len(df):
            try:
                score = silhouette_score(X_pca, df["Cluster"])
                st.metric("Silhouette score", f"{score:.3f}")
            except Exception:
                st.caption("Silhouette score could not be calculated for this dataset.")
        st.download_button("Download clustered dataset (CSV)", df.to_csv(index=False).encode("utf-8"), "career_path_clusters.csv", "text/csv")

    with tab2:
        st.subheader("Top keywords for each cluster")
        for cluster_id in sorted(df["Cluster"].unique()):
            center = model.cluster_centers_[cluster_id]
            # Map PCA cluster centre back to original TF-IDF terms approximately via cluster members.
            member_indices = df.index[df["Cluster"] == cluster_id].tolist()
            mean_vector = X[member_indices].mean(axis=0).A1
            top_indices = mean_vector.argsort()[-10:][::-1]
            top_terms = [terms[i] for i in top_indices if mean_vector[i] > 0]
            with st.expander(f"Cluster {cluster_id} — {len(member_indices)} resumes", expanded=True):
                st.write(", ".join(top_terms) if top_terms else "No prominent keywords found.")
                st.caption("Cluster IDs are numerical groups, not automatically verified job titles. Interpret them using these keywords.")

    with tab3:
        st.subheader("Analyze a resume")
        st.write("Paste resume text or upload a PDF, DOCX, or TXT file. The app predicts the closest cluster learned from your dataset.")
        uploaded_resume = st.file_uploader("Upload resume file", type=["pdf", "docx", "txt"], key="resume")
        pasted_text = st.text_area("Or paste resume text here", height=180, placeholder="Skills, education, experience, career objective...")
        if st.button("Analyze resume", type="primary"):
            try:
                resume_text = read_resume(uploaded_resume) if uploaded_resume is not None else pasted_text
                if not resume_text.strip():
                    st.warning("Please upload a resume or paste some text first.")
                else:
                    cleaned = clean_text(resume_text)
                    if not cleaned.strip():
                        st.warning("Not enough readable text found to analyze.")
                    else:
                        vec = vectorizer.transform([cleaned])
                        # Use PCA fitted on the training set; fit a reusable PCA above instead of refitting.
                        pca_model = PCA(n_components=n_components, random_state=42)
                        pca_model.fit(X.toarray())
                        pred = int(model.predict(pca_model.transform(vec.toarray()))[0])
                        st.success(f"Predicted cluster: {pred}")
                        st.caption("This is an experimental grouping, not a confirmed career title or hiring recommendation.")
                        low = resume_text.lower()
                        matched = {category: [skill for skill in skills if re.search(r"(?<!\w)" + re.escape(skill) + r"(?!\w)", low)] for category, skills in CAREER_SKILLS.items()}
                        matched = {category: skills for category, skills in matched.items() if skills}
                        st.write(f"**Resume word count:** {len(resume_text.split())}")
                        st.write("**Matched skills:**")
                        if matched:
                            for category, skills in matched.items():
                                st.write(f"- **{category}:** {', '.join(skills)}")
                            best_category, found = max(matched.items(), key=lambda item: len(item[1]))
                            suggestions = [s for s in CAREER_SKILLS[best_category] if s not in found][:5]
                            st.write(f"**Skills to consider for {best_category}:** {', '.join(suggestions) if suggestions else 'Keep building relevant skills.'}")
                        else:
                            st.write("No skills from the built-in skill dictionary were detected.")
                        member_indices = df.index[df["Cluster"] == pred].tolist()
                        mean_vector = X[member_indices].mean(axis=0).A1
                        top_indices = mean_vector.argsort()[-10:][::-1]
                        st.write("**Keywords associated with this cluster:** " + ", ".join(terms[i] for i in top_indices if mean_vector[i] > 0))
            except Exception as e:
                st.error(f"Could not analyze the resume: {e}")
except Exception as e:
    st.error(f"Could not load/train the dataset: {e}")
    st.markdown("Check that the CSV has a `Resume` column or `skills` and/or `career_objective` columns, and that it contains at least 3 non-empty rows.")
