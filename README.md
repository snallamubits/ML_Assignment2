# Machine Learning Classification Models – Student Performance Dataset

## a. Problem Statement

The objective of this project is to design, implement, and evaluate multiple
machine learning classification models on a real-world educational dataset.
The goal is to predict student performance levels based on demographic,
social, and school-related attributes, and to compare the performance of
different classification algorithms using standard evaluation metrics.

---

## b. Dataset Description

The dataset used in this project is the **Student Performance Dataset**
obtained from the UCI Machine Learning Repository.

The dataset represents student achievement in secondary education across
two Portuguese schools and was collected using school reports and
questionnaires. It includes academic, demographic, social, and
school-related features.

Two subjects are originally available in the dataset:
- Mathematics
- Portuguese Language

In this project, the dataset is accessed programmatically using the
`ucimlrepo` Python library (Dataset ID: 320), ensuring reproducibility
and data integrity.

### Dataset Characteristics

- Number of instances: More than 1000
- Number of features: More than 30
- Feature types: Numeric and categorical
- Target attribute: Final grade (G3)

### Target Engineering

The numeric final grade (G3) is converted into a **five-level
classification** problem:

| Class | Grade Range | Description   |
|-------|------------|---------------|
| 0     | 0–9        | Very Poor     |
| 1     | 10–11      | Poor          |
| 2     | 12–13      | Satisfactory  |
| 3     | 14–15      | Good          |
| 4     | 16–20      | Excellent     |

To avoid data leakage, intermediate grades (G1 and G2) are excluded
from the feature set, making the prediction task more realistic and
meaningful.

---

## c. Models Used

The following six classification models were implemented using the same
dataset and train–test split:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (KNN) Classifier
4. Naive Bayes Classifier (Gaussian)
5. Random Forest (Ensemble Model)
6. XGBoost (Ensemble Model)

Gaussian Naive Bayes was selected because the feature set consists of
continuous numerical values after preprocessing, making it more
suitable than Multinomial Naive Bayes, which is designed for discrete
count-based data.

---

## d. Evaluation Metrics

Each model was evaluated using the following metrics:

- Accuracy
- AUC Score (One-vs-Rest, Macro Average)
- Precision (Macro Average)
- Recall (Macro Average)
- F1 Score (Macro Average)
- Matthews Correlation Coefficient (MCC)

These metrics provide a balanced evaluation of model performance,
especially for multi-class classification problems with potential
class imbalance.

---

## e. Model Comparison Table

| ML Model                 | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|--------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression      | 0.4154   | 0.6857 | 0.3827    | 0.3762 | 0.3775 | 0.2433 |
| Decision Tree            | 0.3154   | 0.5612 | 0.3102    | 0.2980 | 0.2995 | 0.1228 |
| KNN                      | 0.3000   | 0.5750 | 0.3094    | 0.2922 | 0.2964 | 0.0874 |
| Naive Bayes              | 0.2385   | 0.6098 | 0.3802    | 0.3138 | 0.2145 | 0.1673 |
| Random Forest (Ensemble) | 0.3538   | 0.6711 | 0.3329    | 0.3123 | 0.3131 | 0.1498 |
| XGBoost (Ensemble)       | 0.3000   | 0.6190 | 0.2667    | 0.2684 | 0.2646 | 0.0535 |


---

## f. Observations on Model Performance

| ML Model                 | Observation                                                                                                                                            |
|--------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| Logistic Regression      | Achieved the highest accuracy among all six models. Its linear decision boundary generalised well on this dataset once intermediate grades were removed, outperforming more complex models that struggled with the reduced feature signal. |
| Decision Tree            | Captured feature interactions effectively; however, its performance showed fluctuation across runs, indicating a tendency to overfit the training data without pruning. |
| KNN                      | Produced moderate results but proved sensitive to feature scaling and the choice of neighbourhood size, which affected prediction stability across different sample distributions. |
| Naive Bayes              | Executed very quickly but collapsed most predictions into a single dominant class, resulting in the lowest accuracy. The strong feature-independence assumption and class imbalance in predictions caused a significant gap between precision and F1 score. |
| Random Forest (Ensemble) | Demonstrated solid performance as the second-best model, benefiting from ensemble averaging to reduce variance. However, without the intermediate grade features, the ensemble could not fully leverage its capacity for complex pattern learning. |
| XGBoost (Ensemble)       | Performed below expectation on this dataset, tying with KNN on accuracy and recording the lowest MCC score. The removal of G1 and G2 significantly reduced the predictive signal, limiting the boosting algorithm's ability to learn meaningful patterns from demographic features alone. |

---

## g. Repository Structure

```
project-folder/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
│
└── model/
    ├── __init__.py         # Package initialiser
    ├── dataprep.py         # Data loading and preprocessing
    └── metrics.py          # Model definitions and evaluation
```

---

## h. Execution Instructions

### Local Execution

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## i. Deployment

The Streamlit application is deployed using Streamlit Community Cloud
and provides an interactive interface to:

- Upload custom test data (CSV)
- Select individual models from a dropdown
- View the full model comparison metrics table
- Inspect per-model confusion matrices and classification reports
- Review qualitative observations for each model

---

## j. Conclusion

This project demonstrates a comparative analysis of multiple
classification models on a real-world educational dataset. Contrary to
the general expectation that ensemble methods outperform simpler models,
Logistic Regression achieved the highest accuracy on this dataset. This
outcome is explained by the deliberate exclusion of intermediate grade
features (G1 and G2) to prevent data leakage, which substantially
reduced the predictive signal available. With primarily demographic and
social features remaining, a linear model generalised more effectively
than complex ensemble methods that require richer feature interactions to
demonstrate their advantage. The results highlight that model selection
must account for the nature and richness of available features, and that
simpler models can outperform complex ones when the underlying signal is
limited.
