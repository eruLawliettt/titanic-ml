from omegaconf import OmegaConf

import pandas as pd

from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor
from src.models import get_model
from src.validation import evaluate_model
from src.catboost_validation import evaluate_catboost
from src.dnn_validation import evaluate_dnn, train_final_dnn

def evaluate_configured_model(model_name, model_cfg, X, y, config,):
    
    if model_name == 'dnn':
        
        result = evaluate_dnn(
            X=X,
            y=y,
            n_splits=config.cv.n_splits,
            epochs=model_cfg.training.epochs,
            batch_size=model_cfg.training.batch_size,
            lr=model_cfg.training.lr,
            seed=config.general.seed,
            
            use_family_size=config.preprocessing.use_family_size,
            use_fare_log=config.preprocessing.use_fare_log,
            use_title=config.preprocessing.use_title,
            use_is_alone=config.preprocessing.use_is_alone,
            use_family_category=config.preprocessing.use_family_category,
        )
        
        return result
    
    model_params = OmegaConf.to_container(
        model_cfg.params,
        resolve=True,
    )
    
    if model_name == 'catboost':
        
        cat_features = get_cat_features(config)
        
        result = evaluate_catboost(
            X=X,
            y=y,
            cat_features=cat_features,
            model_params=model_params,
            n_splits=config.cv.n_splits,
            seed=config.general.seed,
            
            use_family_size=config.preprocessing.use_family_size,
            use_title=config.preprocessing.use_title,
            use_fare_log=config.preprocessing.use_fare_log,
            use_is_alone=config.preprocessing.use_is_alone,
            use_family_category=config.preprocessing.use_family_category,
        )
        
        return result
    
    preprocessor = get_preprocessor(model_cfg.model_type, config)
    
    model = get_model(
        model_name=model_name,
        seed=config.general.seed,
        **model_params,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    return evaluate_model(
        estimator=pipeline,
        X=X,
        y=y,
        n_splits=config.cv.n_splits,
        scoring=config.cv.scoring,
        seed=config.general.seed,
    )
    
def main():
    
    config = OmegaConf.load('config/config.yaml')
    
    if (
        config.preprocessing.use_is_alone 
        or config.preprocessing.use_family_category
    ) and not config.preprocessing.use_family_size:
        raise ValueError(
            "use_is_alone and use_family_category require use_family_size=True"
        )
    
    train = pd.read_csv(config.paths.train_path)
    
    X = train.drop(columns='Survived')
    y = train['Survived']
    
    results = []
    
    for model_name, model_cfg in config.models.items():
        if not model_cfg.enabled:
            continue
        
        print(f'\nRunning: {model_name}')

        result = evaluate_configured_model(
            model_name=model_name,
            model_cfg=model_cfg,
            X=X,
            y=y,
            config=config,
        )
        
        results.append({
            "model": model_name,
            "cv_mean": result["mean"],
            "cv_std": result["std"],
        })
    
    if not results:
        raise ValueError("No models are enabled in config.")
    
    results_df = pd.DataFrame(results)
    
    results_df = results_df.sort_values(
        "cv_mean",
        ascending=False,
    ).reset_index(drop=True)
    
    results_df.to_csv(
        config.paths.results_path,
        index=False,
    )
    
    print("\nResults:")
    print(results_df)

    best_model_name = results_df.iloc[0]["model"]
    best_score = results_df.iloc[0]["cv_mean"]

    print(f"\nBest model: {best_model_name}")
    print(f"Best CV score: {best_score:.4f}")
    
    test = pd.read_csv(config.paths.test_path)
    
    best_model_cfg = config.models[best_model_name]
    
    train_best_and_submit(
        X=X,
        y=y,
        test=test,
        best_model_name=best_model_name,
        best_model_cfg=best_model_cfg,
        config=config,
    )
    
    print(
        f"\nSubmission saved to: "
        f"{config.paths.submission_path}"
    )
    
def train_best_and_submit(
    X, y, test, best_model_name, best_model_cfg, config):
    
    if best_model_name == 'catboost':
        train_catboost_and_submit(
            X=X,
            y=y,
            test=test,
            model_cfg=best_model_cfg,
            config=config,
        )
        return
    
    if best_model_name == 'dnn':
        train_dnn_and_submit(
            X=X,
            y=y,
            test=test,
            model_cfg=best_model_cfg,
            config=config,
        )
        return

    train_sklearn_and_submit(
        X=X,
        y=y,
        test=test,
        model_name=best_model_name,
        model_cfg=best_model_cfg,
        config=config,
    )        
    
   
def train_sklearn_and_submit(
    X, y, test, model_name, model_cfg, config):
    
    model_params = OmegaConf.to_container(
        model_cfg.params,
        resolve=True,
    )
    
    preprocessor = get_preprocessor(model_cfg.model_type, config)
    
    model = get_model(
        model_name=model_name,
        seed=config.general.seed,
        **model_params,
    )
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    pipeline.fit(X, y)

    predictions = pipeline.predict(test)

    save_submission(
        test=test,
        predictions=predictions,
        path=config.paths.submission_path,
    )

def train_catboost_and_submit(
    X, y, test, model_cfg, config):
    
    model_params = OmegaConf.to_container(
        model_cfg.params,
        resolve=True,
    )
    
    preprocessor = get_preprocessor('catboost', config)
    
    X_processed = preprocessor.fit_transform(X)
    test_processed = preprocessor.transform(test)

    cat_features = get_cat_features(config)
        
    model = get_model(
        model_name="catboost",
        seed=config.general.seed,
        **model_params,
    )

    model.fit(
        X_processed,
        y,
        cat_features=cat_features,
    )

    predictions = model.predict(test_processed).astype(int)

    save_submission(
        test=test,
        predictions=predictions,
        path=config.paths.submission_path,
    )

def train_dnn_and_submit(
    X, y, test, model_cfg, config):
    
    preprocessor = get_preprocessor('standard', config)
    
    X_processed = preprocessor.fit_transform(X)
    test_processed = preprocessor.transform(test)

    probabilities = train_final_dnn(
        X_train=X_processed,
        y_train=y.to_numpy(),
        X_test=test_processed,
        epochs=model_cfg.training.epochs,
        batch_size=model_cfg.training.batch_size,
        lr=model_cfg.training.lr,
        seed=config.general.seed,
    )

    predictions = (probabilities >= model_cfg.training.threshold).astype(int)

    save_submission(
        test=test,
        predictions=predictions,
        path=config.paths.submission_path,
    )

def get_cat_features(config):
    cat_features = ["Sex", "Embarked", "Pclass"]
    
    if config.preprocessing.use_title:
        cat_features.append("Title")

    if config.preprocessing.use_family_category:
        cat_features.append("FamilyCategory")
        
    return cat_features

def get_preprocessor(model_type, config):
    return build_preprocessor(
        model_type=model_type,
        use_family_size=config.preprocessing.use_family_size,
        use_fare_log=config.preprocessing.use_fare_log,
        use_title=config.preprocessing.use_title,
        use_is_alone=config.preprocessing.use_is_alone,
        use_family_category=config.preprocessing.use_family_category,
    )

def save_submission(test, predictions, path):
    
    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions.astype(int),
    })

    submission.to_csv(
        path,
        index=False,
    )

if __name__ == "__main__":
    main()