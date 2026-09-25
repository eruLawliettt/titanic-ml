from sklearn.linear_model import LogisticRegression

def get_model(model_name: str, seed: int = 42, **model_params):
    """Create and return a model by name."""

    if model_name == "logistic_regression":
        return LogisticRegression(
            random_state=seed,
            **model_params,
        )

    raise ValueError(f"Unknown model: {model_name}")