import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
    
def build_preprocessor():
    """Build the Titanic preprocessing pipeline."""
    
    numerical_features = [
        'Age',
        'Fare',
        'SibSp',
        'Parch',
    ]
    
    categorical_features = [
        'Sex',
        'Embarked',
        'Pclass',
    ]
    
    binary_features = [
        'HasCabin',
    ]
    
    feature_pipeline = Pipeline([
        ('age_imputer', AgeImputer()),
        ('embarked_imputer', EmbarkedImputer()),
        ('has_cabin', HasCabinTransformer()),
    ])
    
    column_transformer = ColumnTransformer(
        transformers=[
            (
                'num',
                StandardScaler(),
                numerical_features,
            ),
            (
                'cat',
                OneHotEncoder(
                    handle_unknown='ignore',
                    sparse_output=False,
                ),
                categorical_features,
            ),
            (
                'binary',
                'passthrough',
                binary_features,
            ),
        ],
        remainder='drop',
    )
    
    return Pipeline([
        ('features', feature_pipeline),
        ('columns', column_transformer),
    ])