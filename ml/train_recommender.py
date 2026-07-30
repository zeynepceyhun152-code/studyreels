"""
StudyReels — Personalized Feed Recommender Training

WHAT THIS TRAINS:
A small logistic regression that predicts P(engagement) for a (user, reel) pair,
so the backend can rank candidate reels for a student's feed.

"Engagement" on one interaction = the student watched >= 6s, OR liked it, OR
answered its quiz correctly. This is our label.

FEATURES (all causal — computed only from a user's PAST interactions, so no leakage):
  1. subj_hist_rate     - user's historical engagement rate within this reel's subject (0.5 if none yet)
  2. type_hist_rate     - user's historical engagement rate with this reel format (pov/ai_footage/avatar)
  3. overall_engage_rate- user's overall engagement rate across everything so far
  4. subj_view_count_norm - log-scaled count of prior views in this subject (confidence signal)
  5. novelty            - 1 if the student has never seen this exact reel id before

We simulate a population of synthetic students, each with a hidden "true" affinity
per subject and per reel-type, and generate interactions from those affinities with
noise. The model never sees the hidden affinities directly — only the same rolling
history features the backend will have access to in production. That's the honest
part: it has to learn to use history as a proxy for taste, same as it would from
real logged activity once the app is live.
"""

import json
import random
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

random.seed(42)
np.random.seed(42)

SUBJECTS = ["Math", "Biology", "Humanities", "STEM"]
TYPES = ["pov", "ai_footage", "avatar"]
N_STUDENTS = 400
INTERACTIONS_PER_STUDENT = 70

rows = []

for student_id in range(N_STUDENTS):
    # Hidden ground-truth taste — the model never sees these directly
    subj_affinity = {s: np.random.beta(2, 2) for s in SUBJECTS}
    type_affinity = {t: np.random.beta(2, 2) for t in TYPES}

    # Rolling history the model IS allowed to see (causal, updates as we go)
    subj_hist = {s: [] for s in SUBJECTS}
    type_hist = {t: [] for t in TYPES}
    overall_hist = []
    seen_reel_ids = set()

    for i in range(INTERACTIONS_PER_STUDENT):
        subject = random.choice(SUBJECTS)
        rtype = random.choice(TYPES)
        reel_id = f"{subject}-{rtype}-{random.randint(1, 12)}"  # finite pool -> repeats happen -> novelty varies

        subj_hist_rate = np.mean(subj_hist[subject]) if subj_hist[subject] else 0.5
        type_hist_rate = np.mean(type_hist[rtype]) if type_hist[rtype] else 0.5
        overall_rate = np.mean(overall_hist) if overall_hist else 0.5
        subj_view_count_norm = min(np.log1p(len(subj_hist[subject])) / np.log1p(15), 1.0)
        novelty = 0.0 if reel_id in seen_reel_ids else 1.0

        # Ground truth engagement probability from HIDDEN affinities + a novelty bonus + noise
        true_p = 0.05 + 0.65 * subj_affinity[subject] + 0.2 * type_affinity[rtype] + 0.1 * novelty
        true_p = np.clip(true_p, 0.02, 0.98)
        engaged = 1 if np.random.rand() < true_p else 0

        rows.append({
            "subj_hist_rate": subj_hist_rate,
            "type_hist_rate": type_hist_rate,
            "overall_engage_rate": overall_rate,
            "subj_view_count_norm": subj_view_count_norm,
            "novelty": novelty,
            "engaged": engaged
        })

        subj_hist[subject].append(engaged)
        type_hist[rtype].append(engaged)
        overall_hist.append(engaged)
        seen_reel_ids.add(reel_id)

df = pd.DataFrame(rows)
print(f"Generated {len(df)} interaction rows across {N_STUDENTS} synthetic students")
print(df["engaged"].value_counts(normalize=True).rename("class balance"))

FEATURES = ["subj_hist_rate", "type_hist_rate", "overall_engage_rate", "subj_view_count_norm", "novelty"]
X = df[FEATURES].values
y = df["engaged"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]
print(f"\nTest accuracy: {accuracy_score(y_test, pred):.3f}")
print(f"Test ROC-AUC : {roc_auc_score(y_test, proba):.3f}")

weights = dict(zip(FEATURES, model.coef_[0].tolist()))
intercept = float(model.intercept_[0])

export = {"weights": weights, "intercept": intercept, "features": FEATURES}
print("\n--- COPY THIS INTO server.js (RECOMMENDER_MODEL) ---")
print(json.dumps(export, indent=2))

with open("/home/claude/studyreels/ml/recommender_weights.json", "w") as f:
    json.dump(export, f, indent=2)
