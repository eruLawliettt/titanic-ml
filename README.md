# Titanic Survival Prediction

Machine learning project for the Kaggle Titanic competition.

The goal is to predict whether a passenger survived based on demographic,
ticket and family-related features.

## Environment

- Python 3.11.16

## Project Structure

```text
titanic-ml/
├── config/
│   └── config.yaml
├── data/
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   ├── validation.py
│   ├── catboost_validation.py
│   ├── dnn_validation.py
│   ├── ensemble_validation.py
│   └── ensemble_training.py
├── main.py
├── requirements.txt
└── README.md
```

## Pipeline

The main pipeline includes:
- missing value handling;
- categorical feature encoding;
- numerical feature scaling where required;
- fenature engineerig;
- Stratified K-Fold cross-validation;
- training and comparison of multiple models;
- automatic selection of the best model;
- final training on the full training dataset;
- Kaggle submission generation.


## Feature Engineering
The following engineered features were explored:
```text
- FamilySize
- IsAlone
- Title
- FareLog
- FamilyCategory
```
Feature engineering can be controlled through config/config.yaml.

## Models
The project includes:
```text
- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- CatBoost
- LightGBM
- XGBoost
- Deep Neural Network
```
Ensemble methods were also tested:
```text
- probability averaging;
- hard voting;
- Linear Regression stacking;
- Ridge stacking.
```

## Validation
Models are evaluated using Stratified K-Fold cross-validation.

The number of folds, random seed and scoring metric are defined in
config/config.yaml.

## Results
Example final cross-validation results:


| Model | CV Accuracy |
| - | - |
| LightGBM | 0.8485 |
| XGBoost | 0.8417 |
| CatBoost | 0.8406 |
| DNN | 0.8350 |
| Random Forest | 0.8316 |
| Decision Tree | 0.8227 |
| KNN | 0.8058 |
| Logistic Regression | 0.8002 |
	
| Ensemble | CV Accuracy |
| - | - |
| Ridge Stacking | 0.8484 |
| Hard Voting | 0.8473 |
| Linear Stacking | 0.8462 |
| Averaging | 0.8439 |

The best local CV model was LightGBM.
Kaggle experiments also showed that the model with the best local CV score
does not necessarily achieve the best leaderboard score.

## Kaggle Results

| Model | Public Score |
|---|---:|
| Ridge Stacking | 0.77033 |
| Logistic Regression | 0.76555 |
| Averaging Ensemble | 0.76076 |
| DNN | 0.75837 |
| LightGBM | 0.74401 |

Ridge stacking achieved the best public leaderboard score, despite having almost the same local CV accuracy as LightGBM. This highlights the importance of generalization and model diversity beyond local validation performance.

## Running the Project
Install dependencies:
```bash
pip install -r requirements.txt
```

Run the full pipeline:
```bash
python main.py
```

The script:
1. loads the dataset;
2. evaluates all enabled models;
3. saves the CV results;
4. selects the best model;
5. trains it on the full training dataset;
6. evaluates all enabled ensembles;
7. selects the best ensemble;
8. trains it on the full training dataset;
9. compares it with the best model;
10. generates submission.csv.

Model selection and preprocessing options can be changed in:
config/config.yaml

## Notebooks
The notebooks / directory contains the full experimental workflow:
- EDA
- preprocessing analysis
- classical ML models
- boosting
- hyperparameter tuning
- DNN experiments
- feature engineering
- ensembles
- final submission