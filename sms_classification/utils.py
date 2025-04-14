from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_pipeline(
    model: ClassifierMixin, numerical_features: list[str], text_feature: str
) -> Pipeline:
    """
    Create a pipeline that combines text and numerical features

    Args:
        model_type: selected ML model

    Returns:
        Pipeline object
    """
    # Define preprocessing for numerical features
    numerical_transformer = StandardScaler()

    # Define preprocessing for text features
    # use english stop words here (very common words)
    # might be better to use list from NLTK or so
    # but would optimizer that later
    text_transformer = TfidfVectorizer(stop_words="english")

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_transformer, numerical_features),
            ("text", text_transformer, text_feature),
        ]
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor), 
            ("classifier", model)
        ]
    )

    return pipeline
