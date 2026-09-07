# Heart Disease Patient Outcome Prediction

An end-to-end healthcare data science portfolio project that predicts the presence of heart disease from clinical attributes in the UCI Cleveland dataset.

## Project objective

Build a reproducible classification workflow, compare multiple algorithms, evaluate medically meaningful errors, and communicate results through an interactive dashboard.

## Dataset

- Source: [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
- Subset: Cleveland
- Rows: 303 patient records
- Inputs: 13 demographic and clinical attributes
- Target: `heart_disease` — 0 for absence and 1 for presence (original diagnosis codes 1–4 are combined)

## Workflow

1. Define the problem and evaluation criteria.
2. Load and audit the data.
3. Remove duplicates and convert the target to binary form.
4. Explore outcome prevalence, distributions, group differences, and correlations.
5. Split the data with stratification.
6. Impute, scale, and encode features inside a leakage-safe pipeline.
7. Compare Logistic Regression, KNN, Decision Tree, Random Forest, and Gradient Boosting using five-fold cross-validation.
8. Evaluate the selected model using recall, F1-score, ROC-AUC, confusion matrix, ROC curve, and precision–recall curve.
9. Examine probability-threshold trade-offs and permutation importance.
10. Present findings in a Streamlit dashboard.

## Project structure

```text
heart_disease_prediction_project/
├── Heart_Disease_Prediction.ipynb
├── heart_disease_cleveland.csv
├── app.py
├── README.md
├── REPORT.md
├── requirements.txt
└── .gitignore
```

## Evaluation approach

Accuracy is included but is not sufficient for a health-screening problem. Recall is emphasized because a false negative means failing to flag a patient who has the positive outcome. ROC-AUC and F1-score provide complementary measures, while the confusion matrix makes each error type visible.

## Verified results (random state 42)

Logistic Regression ranked first in five-fold cross-validation with mean ROC-AUC **0.907**. On the 61-record held-out test set it achieved **0.869 accuracy**, **0.929 recall**, **0.867 F1-score**, and **0.958 ROC-AUC**. These figures are promising for a learning project but are not evidence of clinical validity because the dataset is small.

## Key portfolio strengths

- Real clinical dataset with documented provenance
- End-to-end, leakage-safe machine-learning pipeline
- Five-model cross-validation comparison
- Error and decision-threshold analysis
- Model-agnostic feature importance
- Interactive patient-input dashboard
- Clear limitations and responsible-use statement

## Responsible-use statement

This model is an educational demonstration. The dataset is small, historical, and geographically limited. The project is not a medical device and must not be used for diagnosis, treatment, or individual healthcare decisions. Real deployment would require clinical oversight, external validation, calibration, fairness testing, security, privacy safeguards, and regulatory review.

## Author

**Ankan Chowdhury**  
Aspiring Data Analyst | SQL • Power BI • Python • Excel • Tableau
