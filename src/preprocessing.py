import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

class AgeImputer(BaseEstimator, TransformerMixin):
    """Impute missing ages using median age by passenger title."""
    
    def fit(self, X, y=None):
        """Calculate age medians using training data only."""
        
        title = X["Name"].str.extract(r",\s*([^.]*)\.")[0]

        self.age_medians_ = (X.groupby(title)['Age'].median())
        
        self.global_median_ = X['Age'].median()
        
        return self
    
    def transform(self, X):
        """Fill missing ages using previously calculated medians."""
        
        X = X.copy()
        
        title = X["Name"].str.extract(r",\s*([^.]*)\.")[0]
        
        missing_age = X['Age'].isna()
        
        X.loc[missing_age, 'Age'] = (
            title[missing_age]
            .map(self.age_medians_)
            .fillna(self.global_median_)
        )
        
        return X
        
class EmbarkedImputer(BaseEstimator, TransformerMixin):
    """Impute missing Embarked values using the training-data mode."""

    def fit(self, X, y=None):
        """Calculate the most frequent category using training data only."""

        self.most_frequent_ = X["Embarked"].mode()[0]

        return self

    def transform(self, X):
        """Fill missing Embarked values using the learned category."""

        X = X.copy()

        X["Embarked"] = X["Embarked"].fillna(self.most_frequent_)

        return X
        
class HasCabinTransformer(BaseEstimator, TransformerMixin):
    """Create a binary feature indicating whether cabin information is available."""

    def fit(self, X, y=None):
        """Mark the transformer as fitted."""
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()

        X["HasCabin"] = X["Cabin"].notna().astype(int)

        return X
    
class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select specified columns from the dataset."""
    
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        return X[self.columns].copy()

class FamilySizeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()
        X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
        return X

class isAlone(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()
        X["IsAlone"] = (X["FamilySize"] == 1).astype(int)
        return X

class TitleTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()
        X["Title"] = X["Name"].str.extract(r",\s*([^.]*)\.")[0]
        
        common_titles = ['Mr', 'Miss', 'Mrs', 'Master']
        
        X["Title"] = X["Title"].where(
            X['Title'].isin(common_titles),
            'Rare'
        )
        
        return X

class FareLogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()
        X["Fare"] = np.log1p(X["Fare"])
        return X

class FamilyCategoryTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()
        
        X["FamilyCategory"] = pd.cut(
            X["FamilySize"],
            bins=[0, 1, 4, np.inf],
            labels=["Alone", "Small", "Large"]
        )
        
        return X

def build_preprocessor(model_type: str = 'standard',
                       use_family_size: bool = False,
                       use_is_alone: bool = False,
                       use_title: bool = False,
                       use_fare_log: bool = False,
                       use_family_category: bool = False) -> Pipeline:
    """Build the Titanic preprocessing pipeline."""
    
    numerical_features = ['Age', 'Fare', 'SibSp', 'Parch',]
    categorical_features = ['Sex', 'Embarked', 'Pclass',]
    binary_features = ['HasCabin',]
    if use_family_size:
        numerical_features.append('FamilySize')
    if use_is_alone:
        binary_features.append('IsAlone')
    if use_title:
        categorical_features.append('Title')
    if use_family_category:
        categorical_features.append('FamilyCategory')    
    
    
    feature_steps = [
        ('age_imputer', AgeImputer()),
        ('embarked_imputer', EmbarkedImputer()),
        ('has_cabin', HasCabinTransformer()),
    ]
    
    if use_family_size:
        feature_steps.append(('family_size', FamilySizeTransformer()))
        
    if use_is_alone:
        feature_steps.append(('is_alone', isAlone()))

    if use_title:
        feature_steps.append(('title', TitleTransformer()))
        
    if use_fare_log:
        feature_steps.append(('fare_log', FareLogTransformer()))

    if use_family_category:
        feature_steps.append(('family_category', FamilyCategoryTransformer()))

    feature_pipeline = Pipeline(feature_steps)                                          
    
    if model_type == 'catboost':
        selected_features = (
            numerical_features + categorical_features + binary_features
        )
        
        return Pipeline([
            ('features', feature_pipeline),
            ('columns', ColumnSelector(selected_features))
        ])
    
    if model_type == 'standard':
        numerical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])
        
    elif model_type == 'boosting':
        numerical_transformer = SimpleImputer(strategy="median")
    
    else:
        raise ValueError(f'Unknown model_type: {model_type}')
        
    
    column_transformer = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore',sparse_output=False,),categorical_features),
            ('binary', 'passthrough', binary_features),
        ],
        remainder='drop',
    )
       
    return Pipeline([
        ('features', feature_pipeline),
        ('columns', column_transformer),
    ])