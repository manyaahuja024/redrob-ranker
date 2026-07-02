import json
from docx import Document
import pandas as pd
from collections import Counter
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--candidates",
    default="candidates.jsonl",
    help="Path to candidates.jsonl"
)
parser.add_argument(
    "--out",
    default="submission.csv",
    help="Output CSV path"
)
args = parser.parse_args()


candidates = []
with open(args.candidates, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            candidates.append(json.loads(line))
n = len(candidates)



years = [
    c["profile"].get("years_of_experience", 0)
    for c in candidates
]

interesting_titles = [
    "ML Engineer",
    "AI Research Engineer",
    "Data Scientist",
    "Data Engineer",
    "Analytics Engineer",
    "Backend Engineer"
]

retrieval_skills = {
    "Information Retrieval",
    "Recommendation Systems",
    "Semantic Search",
    "Sentence Transformers",
    "Embeddings",
    "Vector Search",
    "Pinecone",
    "FAISS",
    "RAG",
    "Milvus",
    "retrieval",
    "ranking",
    "recommendation",
    "semantic search",
    "vector search",
    "embeddings",
    "faiss",
    "pinecone",
    "milvus",
    "hybrid retrieval",
    "learning-to-rank",
    "bm25"
}

llm_skills = {
    "LLMs",
    "Fine-tuning LLMs",
    "Prompt Engineering",
    "LangChain",
    "QLoRA",
    "Hugging Face Transformers"
}

production_ml_skills = {
    "MLOps",
    "MLflow",
    "Kubeflow",
    "BentoML",
    "Weights & Biases",
    "Feature Engineering"
}

engineering_skills = {
    "Python",
    "Docker",
    "Kubernetes",
    "AWS",
    "GCP",
    "FastAPI",
    "Flask",
    "Kafka",
    "Microservices"
}

negative_skills = {
    "YOLO",
    "OpenCV",
    "Computer Vision",
    "Object Detection",
    "Image Classification",
    "CNN",
    "Speech Recognition",
    "ASR",
    "TTS"
}


title_scores = {

    # Elite
    "Senior Machine Learning Engineer": 57,
    "Staff Machine Learning Engineer": 57,
    "Senior NLP Engineer": 55,
    "Senior AI Engineer": 55,
    "Lead AI Engineer": 55,
    "Recommendation Systems Engineer": 50,
    "Search Engineer": 50,
    "AI Engineer": 45,
    "Applied ML Engineer": 45,
    "Senior Data Scientist": 40,

    # Strong
    "ML Engineer": 30,
    "AI Research Engineer": 28,
    "Data Scientist": 25,
    "Analytics Engineer": 20,
    "Data Engineer": 18,
    "Backend Engineer": 15,

    # Existing
    "Senior Data Engineer": 15,
    "Senior Software Engineer": 12,
    "Software Engineer": 10
}


negative_titles = {
    "HR Manager",
    "Content Writer",
    "Marketing Manager",
    "Sales Executive",
    "Graphic Designer",
    "Operations Manager",
    "Accountant",
    "Customer Support",
    "Business Analyst",
    "Mechanical Engineer",
    "Civil Engineer"
}

tourist_keywords = [
    "online courses",
    "langchain",
    "openai api",
    "grow my ai",
    "exploring ai",
    "ai enthusiast",
    "side projects"
]
product_industries = {
    "Software",
    "SaaS",
    "AI/ML",
    "Fintech",
    "Food Delivery",
    "E-commerce"
}


evaluation_terms = [
    "evaluation",
    "eval",
    "offline benchmark",
    "a/b",
    "ndcg",
    "mrr",
    "map"
]


def score_candidate(candidate):

    score = 0

    profile = candidate["profile"]

    # Title
    title = profile.get("current_title", "")
    score += title_scores.get(title, 0)

    if title in negative_titles:
        score -= 30

    # Experience
    exp = profile.get("years_of_experience", 0)

    if 6 <= exp <= 8:
        score += 20
    elif 5 <= exp <= 9:
        score += 15
    elif 4 <= exp <= 10:
        score += 10
    elif exp > 12:
        score -= 10

    # Skills-
    skills = {s["name"] for s in candidate["skills"]}

    score += 3 * len(skills & retrieval_skills)
    score += 3 * len(skills & llm_skills)
    score += 2 * len(skills & production_ml_skills)
    score += 1 * len(skills & engineering_skills)

    score -= 2 * len(skills & negative_skills)

    # AI Tourist Penalty
    summary = profile.get("summary", "").lower()

    matches = sum(
        kw in summary
        for kw in tourist_keywords
    )

    if matches >= 3:
        score -= 20

    # Product Company Boost
    product_count = 0

    for job in candidate["career_history"]:
        if job.get("industry") in product_industries:
            product_count += 1

    score += product_count * 2

    # Retrieval/Ranking Experience
    career_text = " ".join(
        job.get("description", "")
        for job in candidate["career_history"]
    ).lower()

    #Evaluation
    for term in evaluation_terms:
        if term in career_text:
            score += 10

    important_terms = [
        "recommendation",
        "retrieval",
        "ranking",
        "search",
        "matching",
        "relevance"
    ]
    for term in important_terms:
        if term in career_text:
            score += 8
    return score


#BM25
doc = Document("job_description.docx")
job_text = "\n".join(
    paragraph.text
    for paragraph in doc.paragraphs
)


candidate_documents = []
for candidate in candidates:
    profile = candidate["profile"]
    headline = profile.get("headline", "")
    summary = profile.get("summary", "")
    title = profile.get("current_title", "")
    industry = profile.get("current_industry", "")
    skills = " ".join(
        skill["name"]
        for skill in candidate["skills"]
    )
    history = " ".join(
        job.get("description", "")
        for job in candidate["career_history"]
    )
    document = " ".join([
        title,
        headline,
        summary,
        skills,
        history,
        industry
    ])
    candidate_documents.append(document.lower())


tokenized_docs = [
    doc.split()
    for doc in candidate_documents
]
bm25 = BM25Okapi(tokenized_docs)
query = job_text.lower().split()
bm25_scores = bm25.get_scores(query)


top10 = sorted(
    enumerate(bm25_scores),
    key=lambda x: x[1],
    reverse=True
)[:10]


bm25_scores = np.array(bm25_scores)
bm25_scores = (
    bm25_scores - bm25_scores.min()
) / (
    bm25_scores.max() - bm25_scores.min()
)

#Tf-Idf
documents = [job_text.lower()] + candidate_documents

vectorizer = TfidfVectorizer(
    stop_words="english"
)
tfidf_matrix = vectorizer.fit_transform(documents)


job_vector = tfidf_matrix[0]
candidate_vectors = tfidf_matrix[1:]
tfidf_scores = cosine_similarity(
    job_vector,
    candidate_vectors
).flatten()


tfidf_scores = (
    tfidf_scores - tfidf_scores.min()
) / (
    tfidf_scores.max() - tfidf_scores.min()
)

#Honeyplot handling
def consistency_penalty(candidate):
    penalty = 0
    profile = candidate["profile"]
    exp = profile.get("years_of_experience", 0)

    # 1. Claimed experience vs career history
    total_months = sum(
        job.get("duration_months", 0)
        for job in candidate["career_history"]
    )
    actual_exp = total_months / 12

    if abs(actual_exp - exp) > 2:
        penalty += 20

    # 2. Skill duration cannot greatly exceed experience
    impossible_skills = 0

    for skill in candidate["skills"]:

        years = skill.get("duration_months", 0) / 12

        if years > exp + 1:
            impossible_skills += 1

    penalty += impossible_skills * 5

    # 3. Too many expert skills is suspicious
    expert_count = sum(
        skill.get("proficiency", "").lower() == "expert"
        for skill in candidate["skills"]
    )
    if expert_count > 8:
        penalty += (expert_count - 8) * 2

    # 4. Senior title but beginner summary
    title = profile.get("current_title", "").lower()
    summary = profile.get("summary", "").lower()
    if "senior" in title:
        beginner_phrases = [
            "online courses",
            "learning ai",
            "building competence",
            "transitioning toward ai",
            "grow my ai",
            "exploring ai",
            "new to ai"
        ]
        if any(p in summary for p in beginner_phrases):
            penalty += 25

    # 5. Recommendation/Search title without retrieval terms
    if any(
        word in title
        for word in [
            "recommendation",
            "search",
            "nlp",
            "machine learning",
            "ai engineer"
        ]
    ):
        career_text = " ".join(
            job.get("description", "")
            for job in candidate["career_history"]
        ).lower()
        retrieval_terms = [
            "retrieval",
            "ranking",
            "recommendation",
            "embeddings",
            "faiss",
            "pinecone",
            "semantic search",
            "vector search",
            "bm25",
            "learning-to-rank",
            "llm",
            "transformer"
        ]
        hits = sum(
            term in career_text
            for term in retrieval_terms
        )
        if hits < 2:
            penalty += 20
    return penalty


def behavioral_score(candidate):
    signals = candidate["redrob_signals"]
    score = 0
    if signals["open_to_work_flag"]:
        score += 5
    score += 6 * signals["recruiter_response_rate"]
    score += 8 * signals["interview_completion_rate"]
    score += min(
        signals["saved_by_recruiters_30d"],
        10
    )
    notice = signals["notice_period_days"]
    if notice <= 30:
        score += 5
    elif notice <= 60:
        score += 2
    return score


def final_score(candidate, bm25, tfidf):
    rule = score_candidate(candidate)
    behavior = behavioral_score(candidate)
    penalty = consistency_penalty(candidate)
    score = (
        rule
        + behavior
        + 35 * bm25
        + 25 * tfidf
        - penalty
    )
    return score


results = []
for i, candidate in enumerate(candidates):

    score = final_score(
        candidate,
        bm25_scores[i],
        tfidf_scores[i]
    )
    results.append(
        {
            "candidate_id": candidate["candidate_id"],
            "score": score,
            "candidate": candidate
        }
    )


results = sorted(
    results,
    key=lambda x: x["score"],
    reverse=True
)


top100 = results[:100]


def build_reasoning(candidate):

    profile = candidate["profile"]
    signals = candidate["redrob_signals"]
    skills = {
        s["name"]
        for s in candidate["skills"]
    }

    reasons = []

    # Title
    title = profile["current_title"]
    if title in title_scores:
        reasons.append(title)

    # Experience
    exp = profile["years_of_experience"]
    reasons.append(f"{exp:.1f} yrs")

    # Core AI skills
    core = len(skills & retrieval_skills)
    if core:
        reasons.append(f"{core} retrieval skills")
    llm = len(skills & llm_skills)
    if llm:
        reasons.append(f"{llm} LLM skills")

    # Behavioral
    if signals["open_to_work_flag"]:
        reasons.append("Open to work")
    if signals["recruiter_response_rate"] > 0.7:
        reasons.append("High recruiter response")
    if signals["interview_completion_rate"] > 0.8:
        reasons.append("Strong interview completion")
    return ", ".join(reasons)


submission = pd.DataFrame({
    "candidate_id": [
        r["candidate_id"]
        for r in top100
    ],
    "rank": range(1,101),
    "score": [
        round(r["score"],6)
        for r in top100
    ],
    "reasoning": [
        build_reasoning(r["candidate"])
        for r in top100
    ]

})

submission.to_csv(
    args.out,
    index=False
)
