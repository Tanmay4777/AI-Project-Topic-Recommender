"""
AI Project Topic Recommender - API Backend
Task: AI-SS-003 | Data Alcott Systems | Student: Tanmay Sah (DAS008051)
"""

from flask import Flask, jsonify, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

app = Flask(__name__)

# Project Topic In-Memory Dataset (Pandas DataFrame)
PROJECTS_DF = pd.DataFrame([
    {
        "id": "PRJ-NLP-01",
        "title": "Conversational AI & Intent Classification Engine",
        "domain": "NLP",
        "difficulty": "Intermediate",
        "keywords": "python nlp transformers bert pytorch chatbot intent classification conversational entity dialog"
    },
    {
        "id": "PRJ-GEN-02",
        "title": "Retrieval-Augmented Generation (RAG) Documentation Agent",
        "domain": "Generative AI",
        "difficulty": "Advanced",
        "keywords": "python langchain llm rag generative openai qdrant vector embeddings dense retrieval"
    },
    {
        "id": "PRJ-NLP-03",
        "title": "Aspect-Based Sentiment Analyzer for Product Reviews",
        "domain": "NLP",
        "difficulty": "Beginner",
        "keywords": "python nltk scikit-learn tf-idf pandas sentiment analysis reviews text classification"
    },
    {
        "id": "PRJ-CV-04",
        "title": "Real-time Defect Detection using YOLOv8 & PyTorch",
        "domain": "Computer Vision",
        "difficulty": "Intermediate",
        "keywords": "python opencv pytorch yolov8 cnn computer vision object detection image segmentation"
    },
    {
        "id": "PRJ-ML-05",
        "title": "Predictive Customer Churn Analysis & Explainable AI",
        "domain": "Machine Learning",
        "difficulty": "Beginner",
        "keywords": "python pandas scikit-learn shap xgboost machine learning churn prediction analytics tabular"
    }
])

def preprocess_text(text: str) -> str:
    """Preprocess text: lowercasing, non-alphanumeric removal, token cleaning."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s-]", "", text)
    return text.strip()

@app.route("/api/recommend_topics", methods=["POST"])
def recommend_topics():
    data = request.get_json() or {}
    user_skills = data.get("skills", "Python NLP Transformers Chatbot PyTorch")
    difficulty = data.get("difficulty", "All")
    top_n = int(data.get("top_n", 5))

    clean_user_input = preprocess_text(user_skills)
    
    # Text Preprocessing & TF-IDF Vectorization
    corpus = [clean_user_input] + PROJECTS_DF["keywords"].apply(preprocess_text).tolist()
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Cosine Similarity Matching
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    results = []
    for idx, score in enumerate(similarities):
        row = PROJECTS_DF.iloc[idx]
        if difficulty != "All" and row["difficulty"] != difficulty:
            continue

        results.append({
            "id": row["id"],
            "title": row["title"],
            "domain": row["domain"],
            "difficulty": row["difficulty"],
            "cosine_similarity": round(float(score) * 100, 2)
        })

    # Top-N Rank
    results.sort(key=lambda x: x["cosine_similarity"], reverse=True)

    return jsonify({
        "status": "success",
        "student_id": "DAS008051",
        "task_id": "AI-SS-003",
        "top_recommendations": results[:top_n]
    })

if __name__ == "__main__":
    app.run(port=5000, debug=True)