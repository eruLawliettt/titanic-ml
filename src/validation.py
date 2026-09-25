from sklearn.model_selection import StratifiedKFold, cross_val_score

def evaluate_model(estimator, X, y, n_splits: int = 5, 
                   scoring: str = 'accuracy', seed: int = 42,
                   n_jobs = 1):
    """Evaluate an estimator using stratified cross-validation."""
    
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )
    
    scores = cross_val_score(
        estimator=estimator,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        error_score='raise'
    )
    
    return {
        'scores': scores,
        'mean': scores.mean(),
        'std': scores.std(),
    }