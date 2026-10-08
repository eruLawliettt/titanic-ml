import numpy as np

from omegaconf import OmegaConf

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression, Ridge

from src.models import get_model
from src.dnn_validation import train_one_fold
from src.pipeline_utils import (
    get_preprocessor,
    get_cat_features,
)


def get_cv(n_splits=5, seed=42):
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    
def get_lgbm_oof(X, y, model_params, config):
    
    cv = get_cv(
        n_splits=config.cv.n_splits,
        seed=config.general.seed,
    )
    
    oof = np.zeros(len(X))
    
    for train_idx, val_idx in cv.split(X, y):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]

        preprocessor = get_preprocessor(config.models.lightgbm.model_type, config)
        
        model = get_model(
            'lightgbm',
            seed=config.general.seed,
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


def get_catboost_oof(X, y, model_params, cat_features, config):
    
    cv = get_cv(
        n_splits=config.cv.n_splits,
        seed=config.general.seed,
    )
    
    oof = np.zeros(len(X))
    
    for train_idx, val_idx in cv.split(X, y):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        
        preprocessor = get_preprocessor('catboost', config)
        
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        
        model = get_model(
            model_name='catboost',
            seed=config.general.seed,
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


def get_dnn_oof(X, y, model_cfg, config):
    
    cv = get_cv(
        n_splits=config.cv.n_splits,
        seed=config.general.seed,
    )
    
    oof = np.zeros(len(X))
    
    for train_idx, val_idx in cv.split(X, y):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx].to_numpy()
        
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx].to_numpy()
        
        preprocessor = get_preprocessor('standard', config)
        
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        
        probabilities = train_one_fold(
            X_train=X_train_processed,
            y_train=y_train,
            X_val=X_val_processed,
            y_val=y_val,
            epochs=model_cfg.training.epochs,
            batch_size=model_cfg.training.batch_size,
            lr=model_cfg.training.lr,
            seed=config.general.seed,
            return_probs=True,
        )
        
        oof[val_idx] = probabilities
        
    return oof


def evaluate_oof_predictions(predictions, y, n_splits, seed):
    
    cv = get_cv(n_splits=n_splits, seed=seed)
    
    scores = []
    
    for _, val_idx in cv.split(predictions, y):
        y_val = y.iloc[val_idx]
        pred_val = predictions[val_idx]
        
        scores.append(
            accuracy_score(y_val, pred_val)
        )
        
    scores = np.array(scores)
        
    return {
        "mean": scores.mean(),
        "std": scores.std(),
    }
    
def evaluate_meta_model(model, X_meta, y, n_splits, seed, threshold):
    
    cv = get_cv(
        n_splits=n_splits,
        seed=seed,
    )

    scores = []

    for train_idx, val_idx in cv.split(X_meta, y):
        X_train = X_meta[train_idx]
        X_val = X_meta[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        predictions = (predictions >= threshold).astype(int)

        scores.append(accuracy_score(y_val, predictions))
        
    scores = np.array(scores)
    
    return {
        "mean": scores.mean(),
        "std": scores.std(),
    }


def evaluate_ensembles(X, y, config):
    if not config.ensemble.enabled:
        return []
    
    ensemble_models = config.ensemble.models
    
    oof_predictions = {}
    
    if 'lightgbm' in ensemble_models:
        print(f'\nRunning: lightgbm')
        model_cfg = config.models.lightgbm
        
        model_params = OmegaConf.to_container(
            model_cfg.params,
            resolve=True,
        )

        oof_predictions["lightgbm"] = get_lgbm_oof(
            X=X,
            y=y,
            model_params=model_params,
            config=config,
        )
        
    if "catboost" in ensemble_models:
        print(f'\nRunning: catboost')
        model_cfg = config.models.catboost

        model_params = OmegaConf.to_container(
            model_cfg.params,
            resolve=True,
        )

        oof_predictions["catboost"] = get_catboost_oof(
            X=X,
            y=y,
            model_params=model_params,
            cat_features=get_cat_features(config),
            config=config,
        )
        
    if "dnn" in ensemble_models:
        print(f'\nRunning: dnn')
        model_cfg = config.models.dnn

        oof_predictions["dnn"] = get_dnn_oof(
            X=X,
            y=y,
            model_cfg=model_cfg,
            config=config
        )
        
    if not oof_predictions:
        raise ValueError(
            'No models are configured for ensemble.'
        )

    X_meta = np.column_stack(
        list(oof_predictions.values())
    )

    ensemble_results = []
    
    #Averaging
    if config.ensemble.methods.averaging:
        average_proba = np.mean(
            X_meta, 
            axis=1
        )

        average_pred = (
            average_proba >= config.general.threshold
        ).astype(int)

        result = evaluate_oof_predictions(
            predictions=average_pred,
            y=y,
            n_splits=config.cv.n_splits,
            seed=config.general.seed
        )

        ensemble_results.append({
            "model": "ensemble_averaging",
            "cv_mean": result["mean"],
            "cv_std": result["std"],
        })
        
    #Hard Voting
    if config.ensemble.methods.hard_voting:
        
        class_predictions = (
            X_meta >= config.general.threshold
        ).astype(int)

        votes = class_predictions.sum(axis=1)

        majority = class_predictions.shape[1] // 2 + 1

        voting_pred = (
            votes >= majority
        ).astype(int)
        
        result = evaluate_oof_predictions(
            predictions=voting_pred,
            y=y,
            n_splits=config.cv.n_splits,
            seed=config.general.seed,
        )

        ensemble_results.append({
            "model": "ensemble_hard_voting",
            "cv_mean": result["mean"],
            "cv_std": result["std"],
        })
            
    # linear stacking
    if config.ensemble.methods.linear_stacking:
        
        result = evaluate_meta_model(
            model=LinearRegression(),
            X_meta=X_meta,
            y=y,
            n_splits=config.cv.n_splits,
            seed=config.general.seed,
            threshold=config.general.threshold,
        )

        ensemble_results.append({
            "model": "ensemble_linear_stacking",
            "cv_mean": result["mean"],
            "cv_std": result["std"],
        })

    # ridge stacking
    if config.ensemble.methods.ridge_stacking:
        result = evaluate_meta_model(
            model=Ridge(alpha=1.0),
            X_meta=X_meta,
            y=y,
            n_splits=config.cv.n_splits,
            seed=config.general.seed,
            threshold=config.general.threshold,
        )

        ensemble_results.append({
            "model": "ensemble_ridge_stacking",
            "cv_mean": result["mean"],
            "cv_std": result["std"],
        })

    return ensemble_results

def get_cat_features(config):
    cat_features = ["Sex", "Embarked", "Pclass"]
    
    if config.preprocessing.use_title:
        cat_features.append("Title")

    if config.preprocessing.use_family_category:
        cat_features.append("FamilyCategory")
        
    return cat_features