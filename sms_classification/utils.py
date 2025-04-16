import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.metrics import auc, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_pipeline(
    model: ClassifierMixin, numerical_features: list[str], text_feature: str
) -> Pipeline:
    """
    Create a pipeline that combines text and numerical features

    Args:
        model_type: selected ML model
        numerical_features: list of numerical feature names
        text_feature: name of text feature to encode

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
        [("preprocessor", preprocessor), ("classifier", model)]
    )

    return pipeline


def plot_confusion_matrix(
    conf_matrix: np.array,
    labels: list[str] = ["legitimate", "spam"],
    figsize: tuple = (10, 8),
):
    """
    Plot confusion matrix

    Args:
        conf_matrix: Confusion matrix array
        labels: List of class labels
    """
    plt.figure(figsize=(10, 8))

    sb.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Prediction")
    plt.ylabel("True label")

    plt.tight_layout()

    plt.show()


def plot_precision_recall(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    figsize: tuple = (10, 6),
):
    """
    Plot precision-recall curve for the model

    Args:
        pipeline: Fitted pipeline
        X_test: Test features
        y_test: Test labels
    """
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    precision, recall, thresholds = precision_recall_curve(
        y_test, y_pred_proba
    )

    # Calculate AUC
    auc_pr = auc(recall, precision)

    plt.figure(figsize=figsize)
    plt.plot(
        recall,
        precision,
        color="darkblue",
        lw=2,
        label=f"Precision-recall curve (AUC = {auc_pr:.2f})",
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.show()


def get_feature_importance(
    pipeline: Pipeline,
    numerical_features: list[str],
    text_feature: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Calculate permutation importances

    Args:
        pipeline: Fitted pipeline
        X_test: Test features
        y_test: Test labels

    Returns:
        DataFrame: Feature names and their importance scores
    """
    # Get the feature names from the preprocessor
    feature_names = []
    feature_names.extend(
        pipeline.named_steps["preprocessor"][
            text_feature
        ].get_feature_names_out()
    )
    feature_names.extend(numerical_features)

    X_test_trafo = pipeline.named_steps["preprocessor"].transform(X_test)

    # Convert sparse matrix to dense if necessary
    if hasattr(X_test_trafo, "toarray"):
        X_test_trafo = X_test_trafo.toarray()

    results = permutation_importance(
        pipeline.named_steps["classifier"],
        X_test_trafo,
        y_test,
        n_repeats=5,
        random_state=42,
    )

    importances = {
        "feature": feature_names,
        "importance": results.importances_mean,
        "std": results.importances_std,
    }

    return pd.DataFrame(importances)
