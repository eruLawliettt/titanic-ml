from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def get_model(model_name: str, seed: int = 42, **model_params):
    """Create and return a model by name."""

    if model_name == "logistic_regression":
        return LogisticRegression(
            random_state=seed,
            **model_params,
        )
        
    if model_name == 'knn':
        return KNeighborsClassifier(
            **model_params,
        )
    
    if model_name == 'decision_tree':
        return DecisionTreeClassifier(
            random_state=seed,
            **model_params,
        )

    raise ValueError(f"Unknown model: {model_name}")