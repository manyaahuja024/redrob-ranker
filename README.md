# Intelligent Candidate Discovery & Ranking

## Overview

This repository contains my solution for the **Redrob Intelligent Candidate Discovery & Ranking Challenge**.

The ranking system combines:

- Rule-based candidate scoring
- BM25 lexical retrieval
- TF-IDF semantic similarity
- Behavioral signal scoring
- Honeypot detection

The pipeline ranks candidates for the supplied job description and generates the required `submission.csv`.

---

# Repository Structure

```
.
├── candidates.jsonl
├── job_description.docx
├── ranker.py
├── validate_submission.py
├── requirements.txt
├── README.md
├── submission_metadata.yaml
└── submission.csv
```

---

# Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Reproducing the Submission

Run a single command:

```bash
python ranker.py --candidates candidates.jsonl --out submission.csv
```

This generates

```
submission.csv
```

---

# Validate the Submission

```bash
python validate_submission.py submission.csv
```

Expected output:

```
Submission is valid.
```

---

# Methodology

The final ranking score combines five components.

### 1. Rule-Based Candidate Score

Candidate profiles are scored using:

- Current role
- Experience
- AI/ML skills
- Retrieval and ranking technologies
- Industry
- Career history
- Profile summary

The scoring is designed to reward production AI engineering experience rather than simple keyword overlap.

---

### 2. Behavioral Signals

Behavioral information from the Redrob platform is incorporated, including:

- Open-to-work status
- Recruiter response rate
- Interview completion rate
- Recruiter saves
- Notice period

These signals improve ranking quality by prioritizing candidates who are more likely to be available and responsive.

---

### 3. Honeypot Detection

The dataset contains intentionally inconsistent candidate profiles.

A lightweight rule-based detector penalizes suspicious profiles based on unrealistic combinations of experience, skill duration, and proficiency.

---

### 4. BM25 Retrieval

Candidate profile text is indexed using BM25 to measure lexical relevance to the supplied job description.

---

### 5. TF-IDF Similarity

TF-IDF vectors and cosine similarity provide an additional semantic relevance signal between candidate profiles and the job description.

---

# Final Score

The final score is computed as:

```
Final Score =
Rule Score
+ Behavioral Score
+ BM25 Score
+ TF-IDF Score
− Honeypot Penalty
```

Candidates are sorted by decreasing score.

The top 100 candidates are exported.

---

# Dependencies

- Python 3.10+
- NumPy
- Pandas
- scikit-learn
- rank-bm25

---

# Runtime

The ranking pipeline is CPU-only and is designed to satisfy the challenge constraints:

- No GPU
- No internet access
- ≤16 GB RAM
- Ranking completes within the required runtime limit
