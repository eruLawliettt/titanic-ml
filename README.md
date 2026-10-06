# Titanic Survival Prediction

Machine learning project for the Kaggle Titanic competition.

The goal is to predict whether a passenger survived based on demographic,
ticket and family-related features.

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
│   └── ensemble_validation.py
├── main.py
├── requirements.txt
└── README.md

Pipeline
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

Feature Engineering
The following engineered features were explored:
- FamilySize
- IsAlone
- Title
- FareLog
- FamilyCategory
Feature engineering can be controlled through config/config.yaml.

Models
The project includes:
- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- CatBoost
- LightGBM
- XGBoost
- Deep Neural Network
Ensemble methods were also tested:
- probability averaging;
- hard voting;
- Linear Regression stacking;
- Ridge stacking.

Validation
Models are evaluated using Stratified K-Fold cross-validation.

The number of folds, random seed and scoring metric are defined in
config/config.yaml.

Results
Example final cross-validation results:
Model	CV Accuracy
LightGBM	0.8485
XGBoost	0.8417
CatBoost	0.8406
DNN	0.8350
Random Forest	0.8316
Decision Tree	0.8227
KNN	0.8058
Logistic Regression	0.8002

The best local CV model was LightGBM.
Kaggle experiments also showed that the model with the best local CV score
does not necessarily achieve the best leaderboard score.

Running the Project
Install dependencies:
pip install -r requirements.txt

Run the full pipeline:
python main.py

The script:
1. loads the dataset;
2. evaluates all enabled models;
3. saves the CV results;
4. selects the best model;
5. trains it on the full training dataset;
6. generates submission.csv.
Model selection and preprocessing options can be changed in:
config/config.yaml

Notebooks
The notebooks/ directory contains the full experimental workflow:
- EDA
- preprocessing analysis
- classical ML models
- boosting
- hyperparameter tuning
- DNN experiments
- feature engineering
- ensembles
- final submission