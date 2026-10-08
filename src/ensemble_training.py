import numpy as np

from omegaconf import OmegaConf
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge

from src.models import get_model
from src.dnn_validation import train_final_dnn
from src.pipeline_utils import (
    get_preprocessor,
    get_cat_features,
)

from src.ensemble_validation import (
    get_lgbm_oof,
    get_catboost_oof,
    get_dnn_oof,
)

def get_lgbm_test_proba(X, y, test, config):
    model_cfg = config.models.lightgbm

    model_params = OmegaConf.to_container(
        model_cfg.params,
        resolve=True,
    )

    preprocessor = get_preprocessor(
        model_cfg.model_type,
        config,
    )

    model = get_model(
        model_name="lightgbm",
        seed=config.general.seed,
        **model_params,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    pipeline.fit(X, y)

    return pipeline.predict_proba(test)[:, 1]

def get_catboost_test_proba(X, y, test, config):
    model_cfg = config.models.catboost

    model_params = OmegaConf.to_container(
        model_cfg.params,
        resolve=True,
    )

    preprocessor = get_preprocessor(
        "catboost",
        config,
    )

    X_processed = preprocessor.fit_transform(X)
    test_processed = preprocessor.transform(test)

    model = get_model(
        model_name="catboost",
        seed=config.general.seed,
        **model_params,
    )

    model.fit(
        X_processed,
        y,
        cat_features=get_cat_features(config),
    )

    return model.predict_proba(test_processed)[:, 1]

def get_dnn_test_proba(X, y, test, config):
    model_cfg = config.models.dnn

    preprocessor = get_preprocessor(
        "standard",
        config,
    )

    X_processed = preprocessor.fit_transform(X)
    test_processed = preprocessor.transform(test)

    return train_final_dnn(
        X_train=X_processed,
        y_train=y.to_numpy(),
        X_test=test_processed,
        epochs=model_cfg.training.epochs,
        batch_size=model_cfg.training.batch_size,
        lr=model_cfg.training.lr,
        seed=config.general.seed,
    )

def get_ensemble_test_predictions(X, y, test, config):
    predictions = {}

    for model_name in config.ensemble.models:

        if model_name == "lightgbm":
            predictions["lightgbm"] = get_lgbm_test_proba(
                X, y, test, config
            )

        elif model_name == "catboost":
            predictions["catboost"] = get_catboost_test_proba(
                X, y, test, config
            )

        elif model_name == "dnn":
            predictions["dnn"] = get_dnn_test_proba(
                X, y, test, config
            )

        else:
            raise ValueError(
                f"Unsupported ensemble model: {model_name}"
            )

    return predictions

def train_ensemble_and_predict(
    X, y, test, ensemble_name, config):
    
    test_predictions = get_ensemble_test_predictions(
        X=X,
        y=y,
        test=test,
        config=config,
    )
    
    test_meta = np.column_stack(
        list(test_predictions.values())
    )
    
    threshold = config.general.threshold
    
    if ensemble_name == 'ensemble_averaging':
        probs = test_meta.mean(axis=1)
         
        return (probs >= threshold).astype(int)
    
    if ensemble_name == 'ensemble_hard_voting':
        class_predictions = (test_meta >= threshold).astype(int)
        
        if class_predictions.shape[1] % 2 == 0:
            raise ValueError(
                "Hard voting requires an odd number of models "
                "to avoid ties."
            )
            
        votes = class_predictions.sum(axis=1)
        
        majority = class_predictions.shape[1] // 2 + 1
        
        return (votes >= majority).astype(int)
    
    oof_preds = {}
    
    if 'lightgbm' in config.ensemble.models:
        model_cfg = config.models.lightgbm
        
        model_params = OmegaConf.to_container(
            model_cfg.params,
            resolve=True,
        )
        
        oof_preds['lightgbm'] = get_lgbm_oof(
            X=X,
            y=y,
            model_params=model_params,
            config=config,
        )
    
    if 'catboost' in config.ensemble.models:
        model_cfg = config.models.catboost
        
        model_params = OmegaConf.to_container(
            model_cfg.params,
            resolve=True,
        )
        
        oof_preds['catboost'] = get_catboost_oof(
            X=X,
            y=y,
            model_params=model_params,
            cat_features=get_cat_features(config),
            config=config,
        )
        
    if 'dnn' in config.ensemble.models:
        oof_preds['dnn'] = get_dnn_oof(
            X=X,
            y=y,
            model_cfg=config.models.dnn,
            config=config,
        )
        
    X_meta = np.column_stack(list(oof_preds.values()))
    
    if ensemble_name == 'ensemble_linear_stacking':
        meta_model = LinearRegression()
        
    elif ensemble_name == 'ensemble_ridge_stacking':
        meta_model = Ridge(alpha=1.)
        
    else:
        raise ValueError(
            f"Unknown ensemble: {ensemble_name}"
        )
        
    meta_model.fit(X_meta, y)
    
    probs = meta_model.predict(test_meta)
    
    return (probs >= threshold).astype(int)