# Alzheimer's Disease Prediction — Classification Model Comparison

A machine learning project that predicts Alzheimer's disease diagnosis from demographic, lifestyle, clinical and cognitive features. It covers preprocessing, exploratory analysis, and a comparison of four classifiers tuned with cross-validated grid search, followed by an interpretation of the Logistic Regression model.

## Contents

| File | Description |
|------|-------------|
| `AlzheimersDiseasePrediction.ipynb` | Jupyter notebook with the full pipeline: preprocessing, EDA, model training, evaluation and interpretation |
| `AlzheimersDiseasePrediction.pptx` | Presentation slides summarizing the project, with a focus on Logistic Regression |

## Dataset

- **File:** `alzheimers_disease_data.csv` ([Alzheimer's Disease Dataset on Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset))
- **Size:** 2,149 patients × 35 columns
- **Target:** `Diagnosis` (0 = Healthy, 1 = Alzheimer's)
- **Class balance:** 64.6% healthy / 35.4% Alzheimer's
- **Feature groups:**
  - *Demographics:* age, gender, ethnicity, education level
  - *Lifestyle:* BMI, smoking, alcohol consumption, physical activity, diet and sleep quality
  - *Medical history:* family history, cardiovascular disease, diabetes, depression, head injury, hypertension
  - *Clinical measurements:* blood pressure, cholesterol (total, LDL, HDL, triglycerides)
  - *Cognitive & functional assessments:* MMSE, functional assessment, ADL, memory complaints, behavioral problems
  - *Symptoms:* confusion, disorientation, personality changes, difficulty completing tasks, forgetfulness

> The dataset is not included in this repository. Download it and place the CSV in the same folder as the notebook.

## Workflow

1. **Data quality checks** — no missing values and no duplicate rows were found.
2. **Outlier analysis** — IQR method and boxplots for 15 continuous features (no outliers detected).
3. **Scaling exploration** — comparison of standardization (`StandardScaler`, z-score) and min–max normalization (`MinMaxScaler`).
4. **Preprocessing** — dropped identifier columns (`PatientID`, `DoctorInCharge`) and one-hot encoded `Ethnicity`.
5. **Train/test split** — 80/20 stratified split (1,719 / 430 samples) to preserve class ratios.
6. **Model training** — four models trained with `GridSearchCV` using stratified 5-fold cross-validation, optimized for F1 score. Scaling is applied inside a `Pipeline` for scale-sensitive models to prevent data leakage.
7. **Evaluation** — accuracy, precision, recall, F1, ROC-AUC and confusion matrices on the held-out test set.
8. **Interpretation** — analysis of Logistic Regression coefficients and the effect of the regularization parameter `C`.

## Models & Hyperparameter Search

| Model | Scaling | Search space |
|-------|---------|--------------|
| Logistic Regression | StandardScaler | `C`: 0.01, 0.1, 1, 10, 100 |
| K-Nearest Neighbors | StandardScaler | `n_neighbors`: 3–15, `weights`: uniform / distance |
| Decision Tree | — | `max_depth`: 3, 5, 7, 10, None; `min_samples_leaf`: 1, 5, 10 |
| Random Forest | — | `n_estimators`: 100, 300; `max_depth`: 5, 10, None; `min_samples_leaf`: 1, 5 |

## Results (Test Set)

| Model | CV F1 | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|------:|---------:|----------:|-------:|---:|--------:|
| **Decision Tree** | 0.922 | **0.942** | 0.915 | **0.921** | **0.918** | 0.941 |
| Random Forest | 0.900 | 0.935 | **0.949** | 0.862 | 0.903 | **0.943** |
| Logistic Regression | 0.765 | 0.821 | 0.748 | 0.743 | 0.746 | 0.887 |
| KNN | 0.551 | 0.723 | 0.681 | 0.408 | 0.510 | 0.772 |

**Best hyperparameters**
- Decision Tree: `max_depth=5`, `min_samples_leaf=5`
- Random Forest: `n_estimators=300`, `max_depth=10`, `min_samples_leaf=1`
- Logistic Regression: `C=1`
- KNN: `n_neighbors=11`, `weights='uniform'`

### Key Findings

- **Tree-based models perform best.** The Decision Tree achieved the highest F1 and recall, which matters most in a medical setting where missed diagnoses (false negatives) are costly. Random Forest had the highest precision and ROC-AUC.
- **Logistic Regression** offers a solid, interpretable baseline. Its most influential features were:
  - **Lower risk:** higher `FunctionalAssessment` (−1.33), `ADL` (−1.27) and `MMSE` (−0.86) scores
  - **Higher risk:** `MemoryComplaints` (+1.14) and `BehavioralProblems` (+0.93)
  - Most lifestyle and clinical measurements had only small coefficients.
- **KNN performed worst**, likely because many weakly informative features dilute distance-based similarity.
- The `C` sensitivity analysis shows that cross-validation F1 stabilizes around `C = 1`; stronger regularization reduces performance.

## Getting Started

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook AlzheimersDiseasePrediction.ipynb
```

## Tech Stack

Python · pandas · NumPy · scikit-learn · Matplotlib · seaborn · Jupyter

## Disclaimer

This project was developed for educational purposes. The models are not intended for clinical use or medical diagnosis.
