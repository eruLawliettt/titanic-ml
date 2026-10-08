from src.preprocessing import build_preprocessor

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
