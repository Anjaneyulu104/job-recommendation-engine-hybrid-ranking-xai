# Job Recommendation Engine with Hybrid Ranking, Explainability and Feedback-Driven Reranking using NLP

**MSc Artificial Intelligence with Work Experience · Sheffield Hallam University**  
**Researcher:** Venkata Anjaneyulu Medaboina  
**Supervisor:** Jude E. Ameh

An academic prototype that converts a resume profile into a **ranked list of job roles** and explains the evidence behind each recommendation. It combines skill overlap, TF-IDF text similarity, a supervised classification signal, experience fit and education fit. A separate offline example demonstrates how feedback can adjust rankings.

> **Research-use notice:** This is a proof-of-concept decision-support system, **not** a hiring tool. Scores do not establish a candidate's qualifications, suitability for an actual vacancy or employment eligibility.

## 1. Research problem and aim

A single predicted occupation gives job seekers few alternatives and little explanation. Exact keyword matching can also miss useful relationships between the information in a resume and patterns associated with different roles. This project asks:

> To what extent can a hybrid NLP-based job recommendation engine improve recommendation relevance and transparency compared with a basic keyword/TF-IDF matching approach?

The aim is to implement and critically evaluate a reproducible hybrid ranking prototype using an existing synthetic dataset, while exposing the scores and skill evidence used in its recommendations.

## 2. What the project includes

- **Data preparation:** schema checks, handling missing values, numeric experience cleaning, removal of the `Name` field and creation of pseudonymous candidate IDs.
- **Feature engineering:** parsed skills, skill counts, experience bands and a candidate-only text representation.
- **Classification baseline:** TF-IDF with Logistic Regression, compared with a majority-class DummyClassifier.
- **Role matching:** role profiles estimated from training records, skill overlap and TF-IDF/cosine similarity.
- **Hybrid ranking:** five weighted evidence signals, with weights chosen on validation data only.
- **Explanations:** matched skills, missing skills and the score of each component.
- **Evaluation:** held-out ranking metrics, simpler ranking baselines, classifier diagnostics and descriptive subgroup checks.
- **Offline feedback demo:** small positive/negative score changes; the demonstration does **not** retrain the model or use research participants.

The main research implementation is the Jupyter/Kaggle notebook. A separate Streamlit demonstration app can provide a user interface **if `app.py` and its dependencies are included in the repository**.

## 3. Dataset

**Source:** [AI Resume Matcher Dataset – 2000 Samples (Kaggle)](https://www.kaggle.com/datasets/hmnshudhmn24/ai-resume-matcher-dataset-2000-samples)

The study's signed ethics documentation records the dataset as synthetic and its licence as **CC0: Public Domain**. Check the dataset page and its current licence terms before redistributing a copy.

| Property | Value |
| --- | ---: |
| Resume records | 2,000 |
| Original applied-role labels | 8 |
| Parsed skill vocabulary | 21 unique skills |
| Training / validation / test | 1,200 / 400 / 400 |

Required CSV columns:

```text
Name, Experience_Years, Skills, Education, Applied_Job_Role
```

The notebook searches common local and Kaggle locations for a compatible CSV. Its preferred filename is **`large_resume_dataset.csv`**. The `Name` column is removed before modelling; it is not used in the recommendation score.

**Important data limitation:** `Applied_Job_Role` records the role associated with a synthetic resume. It is an *evaluation proxy*, not an independently verified label for the best career or a real vacancy-level match. Some listed skills are weakly associated with the original role labels.

## 4. System architecture

```text
Synthetic resume CSV
        |
        v
Schema checks -> cleaning -> remove Name -> candidate features
        |
        v
Stratified split: 60% train | 20% validation | 20% test
        |
        +--> TRAIN: fit TF-IDF + classifier + role profiles
        |                |
        |                v
        |       Five recommendation signals
        |       - skill overlap
        |       - TF-IDF cosine similarity
        |       - classifier probability
        |       - experience fit
        |       - education fit
        |                |
        +--> VALIDATION: compare candidate weight sets
        |                |
        |                v
        |         Freeze selected weights
        |                |
        +--> TEST: final ranking and baseline evaluation
                         |
                         v
                Ranked Top-N job roles
                + evidence-based explanations
                         |
                         v
                Optional OFFLINE feedback demo
```

**Leakage-control principle:** the original `Applied_Job_Role` is the supervised target. TF-IDF vocabularies and role profiles are fitted using the training split; candidate weights are chosen using validation results. The earlier rule-derived competency target is not used as a supervised target in the final notebook.

### Final hybrid score

The notebook selected the **Skill Focus** configuration using validation **nDCG@5**, with MRR as a secondary selection metric:

```text
Score = 0.40 × skill fit
      + 0.30 × TF-IDF similarity
      + 0.10 × classifier probability
      + 0.15 × experience fit
      + 0.05 × education fit
```

These are research-prototype weights, not validated employment-selection criteria. The offline feedback demonstration uses a **±0.08 adjustment** to illustrate reranking; it is separate from the five base weights.

**Terminology:** the implemented text-similarity component uses **TF-IDF and cosine similarity**. It is not BERT, Sentence-BERT or a transformer embedding model.

## 5. Reproduce the notebook

### Option A — Kaggle

1. Open the final `.ipynb` notebook in Kaggle.
2. Attach the Kaggle dataset linked above, or upload `large_resume_dataset.csv` as a dataset input.
3. Run the notebook cells from top to bottom.
4. Inspect the generated outputs under `/kaggle/working/job_recommendation_final_outputs/`.

### Option B — Local Jupyter

Use Python 3.11 or a compatible environment:

```bash
python -m pip install numpy pandas scikit-learn matplotlib plotly jupyter
python -m jupyter notebook
```

Open the final project notebook, place `large_resume_dataset.csv` in the working directory (or an accessible subdirectory), and run all cells in order. Plotly is optional: where implemented, the notebook falls back to Matplotlib when Plotly is unavailable.

The notebook uses **random seed 42** and creates a local `job_recommendation_final_outputs/` folder outside Kaggle.

### Research outputs

Depending on notebook execution, the output folder contains CSV tables, figures and portable files such as:

```text
job_recommendation_final_outputs/
├── tables/
│   ├── classification_validation_metrics.csv
│   ├── ranking_method_comparison.csv
│   ├── validation_weight_selection.csv
│   ├── final_results_summary.csv
│   └── methodological_integrity_checks.csv
├── figures/
├── project_metadata.json
├── role_profiles_portable.csv
└── skill_vocabulary.json
```

The notebook also creates a ZIP of generated research outputs. Exact contents depend on which cells were executed. JSON/CSV exports document the experiment; they are **not substitutes for a fitted classifier and TF-IDF vectorizer** if creating a separate inference application.

## 6. Optional Streamlit demonstration

**Only use this section if the repository also contains `app.py` and `requirements.txt`.** The app is a demonstration interface and its session-based recommendation settings may differ from the validation-selected research notebook. Do not present app outputs as the notebook's held-out test results.

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app accepts text-based PDF, DOCX and TXT resumes or pasted text and generates role recommendations with score-level explanations. Scanned, image-only PDFs are not supported without OCR. The packaged app trains from the CSV at startup rather than relying on incompatible pre-saved `.joblib` files.

## 7. Measured research results

The following figures are from the final executed research notebook's **400-record held-out test split**, not from a live job-seeker study.

| Evaluation | Metric | Result |
| --- | --- | ---: |
| Original-label classifier | Accuracy | **15.50%** |
| Original-label classifier | Macro F1 | **0.1533** |
| Hybrid ranking | MRR | **0.3595** |
| Hybrid ranking | Recall@1 | **14.50%** |
| Hybrid ranking | Recall@3 | **40.00%** |
| Hybrid ranking | Recall@5 | **63.75%** |
| Hybrid ranking | nDCG@5 | **0.3885** |

The hybrid method had modest advantages on **Recall@5 and nDCG@5** over the tested single-signal ranking baselines. The classifier-only ranking had slightly higher **MRR and Recall@1**. Therefore, the hybrid system did **not** outperform every baseline on every metric.

**Interpretation matters:** there are only eight possible roles, so a uniformly random Top-5 list would contain a designated role on average **5/8 = 62.5%** of the time. The observed 63.75% Recall@5 is only slightly higher than this reference. It measures recovery of the dataset's *applied role*, not verified career suitability. The value of the prototype is its transparent, auditable combination of signals and its critical evaluation, rather than a claim of highly accurate hiring decisions.

## 8. Explainability and feedback

For a candidate-role recommendation, the engine can return the role rank, combined score, each component score, matched skills and missing skills. These are **faithful score-component explanations**, not evidence that the system understands a candidate's abilities or predicts employment success.

The offline feedback example adds **+0.08** or **−0.08** to selected role scores and reranks the results. It does not use participant feedback, save a real user-feedback dataset, or retrain model parameters.

## 9. Ethics and limitations

The research uses secondary synthetic data under the signed UREC2 scope. No interviews, surveys, participant recruitment or live hiring decisions formed part of the study. The direct `Name` field was excluded from modelling. Education and experience are used as structured factors; the notebook's education/experience group summaries are **descriptive diagnostics, not a demonstration of fairness**.

Principal limitations include:

- Sparse synthetic profiles (approximately 2.11 listed skills per record) and a 21-skill vocabulary.
- Weak alignment between original role labels and available resume attributes.
- Only eight available roles and no independent resume–vacancy relevance judgments.
- Heuristic experience/education fit rules and a limited weight-configuration search.
- TF-IDF vector-space matching rather than contextual transformer embeddings.
- Offline feedback adjustment instead of a validated online-learning system.
- No human-beneficiary usefulness study; such a study would require appropriate ethics approval.

**Do not use this prototype to screen, reject, shortlist or rank real applicants in an employment process.**

## 10. Future work

Future research could use independently labelled resume–vacancy pairs, real job descriptions and richer skill representations; compare TF-IDF with sentence-transformer embeddings; investigate learning-to-rank and calibrated feedback; and evaluate perceived usefulness and fairness through a separately approved human-participant study.

## 11. Research and source code

- **Repository:** [job-recommendation-engine-hybrid-ranking-xai](https://github.com/Anjaneyulu104/job-recommendation-engine-hybrid-ranking-xai)
- **Dataset:** [AI Resume Matcher Dataset – 2000 Samples](https://www.kaggle.com/datasets/hmnshudhmn24/ai-resume-matcher-dataset-2000-samples)
- **Primary research artefact:** the final leakage-safe Jupyter/Kaggle `.ipynb` notebook provided in this repository.

The notebook, dataset (subject to its licence terms), experimental outputs and any optional app should be committed only when they are actually included and their contents have been checked. This README describes the notebook workflow; the Streamlit steps are optional.

---

*Academic MSc research prototype · Sheffield Hallam University*
