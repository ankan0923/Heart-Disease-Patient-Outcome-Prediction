# Heart Disease Patient Outcome Prediction — Project Report

## Executive summary

This project develops a classification workflow to estimate whether a patient record shows evidence of heart disease. It uses the Cleveland subset of the UCI Heart Disease dataset and compares five machine-learning algorithms. The purpose is applied learning and portfolio demonstration, not clinical diagnosis.

## Problem statement

Healthcare teams may use risk-screening tools to prioritize patients for further assessment. The analytical goal is to distinguish positive and negative heart-disease outcomes while paying particular attention to false negatives.

## Data and preparation

The dataset contains 303 records, 13 predictor variables, and an original diagnosis field ranging from 0 to 4. Diagnosis 0 is converted to the negative class; values 1–4 form the positive class. Question-mark values are interpreted as missing. Numerical values are median-imputed and standardized; categorical values are mode-imputed and one-hot encoded. All transformations occur inside model pipelines to prevent leakage.

## Analysis and modelling

Exploratory analysis covers class prevalence, numeric distributions, outcome-group box plots, and correlations. Logistic Regression, K-Nearest Neighbours, Decision Tree, Random Forest, and Gradient Boosting are compared using stratified five-fold cross-validation. Model selection considers recall, F1-score, and ROC-AUC rather than accuracy alone.

The final test evaluation includes a classification report, confusion matrix, ROC curve, and precision–recall curve. A threshold analysis shows how changing the probability cut-off affects precision and recall. Permutation importance identifies the inputs that most affect predictive performance without claiming causal relationships.

## Verified model results

| Model | CV accuracy | CV recall | CV F1 | CV ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.847 | 0.819 | 0.831 | 0.907 |
| Random Forest | 0.802 | 0.783 | 0.781 | 0.898 |
| KNN | 0.818 | 0.747 | 0.790 | 0.883 |
| Gradient Boosting | 0.802 | 0.775 | 0.781 | 0.854 |
| Decision Tree | 0.711 | 0.701 | 0.687 | 0.774 |

Logistic Regression was selected because it recorded the highest mean cross-validated ROC-AUC. At the default 0.50 threshold, its held-out test metrics were accuracy **0.869**, precision **0.812**, recall **0.929**, F1-score **0.867**, and ROC-AUC **0.958**. The test set contains only 61 records, so these values should be treated as an illustrative estimate rather than a stable clinical-performance claim.

## Business and clinical interpretation

- Higher recall reduces the number of positive cases missed by the model, but usually increases false alarms.
- Threshold selection is a policy decision and must reflect the cost of each error type.
- Feature importance helps explain model behaviour but does not prove that a variable causes heart disease.
- An interactive dashboard makes aggregate patterns and individual demonstration predictions easier to communicate.

## Limitations

- The sample is small and historical.
- The data comes from a limited clinical setting and may not represent other populations.
- A single train/test experiment cannot establish clinical validity.
- Subgroup fairness, probability calibration, temporal stability, and external validation are required before considering operational use.
- The model cannot replace professional medical judgment.

## Conclusion

The project demonstrates a complete healthcare machine-learning workflow: auditable cleaning, visual exploration, leakage-safe preprocessing, cross-validated model comparison, error analysis, explainability, dashboard communication, and responsible reporting. Its value is educational and methodological rather than diagnostic.

## Recommended next steps

1. Validate the selected model on a larger and more recent external dataset.
2. Assess calibration and report confidence intervals.
3. Compare performance across demographic subgroups.
4. Tune the operating threshold with domain experts.
5. Add monitoring for data drift and model degradation if a governed pilot is approved.
