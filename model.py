# core/model.py
"""
Módulo de modelagem: treinamento e classificação com Random Forest.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 100,
    max_depth: int | None = None,
    random_state: int = 42,
    cv_folds: int = 5,
) -> tuple[RandomForestClassifier, dict]:
    """
    Treina um RandomForestClassifier com validação cruzada opcional.

    Args:
        X: Features (n_samples, n_features)
        y: Labels (n_samples,)
        n_estimators: Número de árvores
        max_depth: Profundidade máxima (None = sem limite)
        random_state: Semente aleatória
        cv_folds: Número de folds para cross-validation (0 = desativado)

    Returns:
        model: Modelo treinado
        metrics: Dicionário com métricas de treinamento
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        oob_score=True,        # Out-of-bag score sem custo extra
        class_weight="balanced",  # Lida melhor com classes desbalanceadas
    )
    model.fit(X, y)

    # Métricas no conjunto de treinamento (referência)
    y_pred_train = model.predict(X)
    metrics = {
        "oob_score": round(model.oob_score_, 4),
        "train_accuracy": round(accuracy_score(y, y_pred_train), 4),
        "n_classes": len(np.unique(y)),
        "n_samples": len(y),
        "n_features": X.shape[1],
        "classification_report": classification_report(y, y_pred_train, output_dict=True),
        "confusion_matrix": confusion_matrix(y, y_pred_train),
    }

    # Cross-validation (desativa se cv_folds <= 1 ou dataset muito pequeno)
    if cv_folds > 1 and len(y) >= cv_folds * 10:
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring="accuracy", n_jobs=-1)
        metrics["cv_mean"] = round(cv_scores.mean(), 4)
        metrics["cv_std"] = round(cv_scores.std(), 4)

    return model, metrics


def classify_image(
    model: RandomForestClassifier,
    X_full: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """
    Classifica todos os pixels da imagem.

    Args:
        model: Modelo treinado
        X_full: Features de todos os pixels (height*width, n_features)
        height: Altura da imagem
        width: Largura da imagem

    Returns:
        classified: Mapa classificado (height, width) int
    """
    prediction = model.predict(X_full)
    return prediction.reshape(height, width).astype(np.int32)


def get_feature_importance(
    model: RandomForestClassifier,
    feature_names: list[str],
) -> dict:
    """
    Retorna importância das features ordenada de forma decrescente.

    Args:
        model: Modelo treinado
        feature_names: Lista com nomes das features

    Returns:
        Dicionário {feature: importância}
    """
    importances = model.feature_importances_
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    return {name: round(float(imp), 4) for name, imp in pairs}
