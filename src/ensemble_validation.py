import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor
from src.models import get_model
from src.dnn_validation import train_one_fold

def get_cv(n_splits=5, seed=42):
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    
def get_lgbm_oof(X, y, model_params, n_splits=5, seed=42):
    cv = get_cv(n_splits=n_splits, seed=seed)
    
    oof = np.zeros(len(X))
    
    for train_idx, val_idx in cv.split(X, y):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        
        X_val = X.iloc[val_idx]

        preprocessor = build_preprocessor(
            model_type='boosting',
            use_family_size=True,
        )
        
        model = get_model(
            'lightgbm',
            seed=seed,
            **model_params,
        )
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model),
        ])
        
        pipeline.fit(X_train, y_train)
        
        probabilities = pipeline.predict_proba(X_val)[:, 1]
        
        oof[val_idx] = probabilities
        
    return oof

def get_catboost_oof(X, y, model_params, cat_features, n_splits=5, seed=42):
    cv = get_cv(n_splits=n_splits, seed=seed)
    
    oof = np.zeros(len(X))
    
    for train_idx, val_idx in cv.split(X, y):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
                
        X_val = X.iloc[val_idx]
        
        preprocessor = build_preprocessor(
            model_type='catboost'
        )
        
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        
        model = get_model(
            model_name='catboost',
            seed=seed,
            **model_params
        )
        
        model.fit(
            X_train_processed,
            y_train,
            cat_features=cat_features
        )
        
        probabilities = model.predict_proba(X_val_processed)[:, 1]
        
        oof[val_idx] = probabilities
        
    return oof

def get_dnn_oof(X, y, n_splits=5, epochs=100, batch_size=32, lr=3e-3, seed=42):
    cv = get_cv(n_splits=n_splits, seed=seed)
    
    oof = np.zeros(len(X))
    
    for train_idx, val_idx in cv.split(X, y):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx].to_numpy()
        
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx].to_numpy()
        
        preprocessor = build_preprocessor(
            model_type='standard',
            use_family_size=True,
            use_fare_log=True
        )
        
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        
        probabilities = train_one_fold(
            X_train=X_train_processed,
            y_train=y_train,
            X_val=X_val_processed,
            y_val=y_val,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            return_probs=True
        )
        
        oof[val_idx] = probabilities
        
    return oof
    
    
    
    