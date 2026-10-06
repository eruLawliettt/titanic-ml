from omegaconf import OmegaConf

import pandas as pd

from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor
from src.models import get_model
from src.validation import evaluate_model

def main():
    config = OmegaConf.load('config/config.yaml')
    
    train = pd.read_csv(config.paths.train_path)
    
    X = train.drop(columns='Survived')
    y = train['Survived']
    
    results = []
    
    for model_name, model_cfg in config.models.items():
        if not model_cfg.enabled:
            continue
        
        print(f'\nRunning: {model_name}')
        
        model_params = OmegaConf.to_container(
            model_cfg.params,
            resolve=True,
        )
        
        preprocessor = build_preprocessor(
            model_type=model_cfg.model_type,
            use_family_size=config.preprocessing.use_family_size,
            use_fare_log=config.preprocessing.use_fare_log,
            use_title=config.preprocessing.use_title,
            use_is_alone=config.preprocessing.use_is_alone,
            use_family_category=config.preprocessing.use_family_category,
        )
        
        model = get_model(
            model_name=model_name,
            seed=config.general.seed,
            **model_params
        )
        
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])
        
        result = evaluate_model(
            estimator=pipeline,
            X=X,
            y=y,
            n_splits=config.cv.n_splits,
            scoring=config.cv.scoring,
            seed=config.general.seed,
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
    
    submission = train_best_and_submit(
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
    
    model_params = OmegaConf.to_container(
        best_model_cfg.params,
        resolve=True,
    )
    
    preprocessor = build_preprocessor(
        model_type=best_model_cfg.model_type,
        use_family_size=config.preprocessing.use_family_size,
        use_fare_log=config.preprocessing.use_fare_log,
        use_title=config.preprocessing.use_title,
        use_is_alone=config.preprocessing.use_is_alone,
        use_family_category=config.preprocessing.use_family_category,
    )
    
    model = get_model(
        model_name=best_model_name,
        seed=config.general.seed,
        **model_params,
    )
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    pipeline.fit(X, y)

    predictions = pipeline.predict(test)
    
    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions.astype(int),
    })

    submission.to_csv(
        config.paths.submission_path,
        index=False,
    )

    return submission
    
if __name__ == "__main__":
    main()