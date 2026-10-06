import numpy as np

from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from src.preprocessing import build_preprocessor

def evaluate_catboost(X, y, cat_features, model_params=None, 
                      n_splits=5, seed=42, early_stopping_rounds=None,
                      use_family_size=False, use_fare_log=False,
                      use_title=False, use_is_alone=False,
                      use_family_category=False,):
    
    if model_params is None:
        model_params = {}
        
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    
    scores = []
    best_iterations = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        
        preprocessor = build_preprocessor('catboost', 
                                          use_family_size=use_family_size, 
                                          use_fare_log=use_fare_log,
                                          use_title=use_title,
                                          use_is_alone=use_is_alone,
                                          use_family_category=use_family_category)
        
        X_train_processed = preprocessor.fit_transform(X_train, y_train)
        X_val_processed = preprocessor.transform(X_val)
        
        model = CatBoostClassifier(
            cat_features=cat_features,
            random_seed=seed,
            verbose=False,
            **model_params,
        )
        
        model.fit(X_train_processed, y_train, 
                  eval_set=(X_val_processed, y_val), 
                  early_stopping_rounds=early_stopping_rounds)
        
        best_iterations.append(model.get_best_iteration())
        
        predictions = model.predict(X_val_processed).ravel()
        
        score = accuracy_score(y_val, predictions)
        scores.append(score)
        
        #print(f'Fold {fold}: {score:.4f}')
        
    scores = np.array(scores)
    best_iterations = np.array(best_iterations)
    
    return {
        "scores": scores,
        "mean": scores.mean(),
        "std": scores.std(),
        "best_iterations": best_iterations,
        "mean_best_iteration": best_iterations.mean(),
    }