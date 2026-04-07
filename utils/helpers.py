# utils/helpers.py
"""
Funções auxiliares usadas pelo app.py: validação, formatação e estatísticas.
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def validate_shapes(image: np.ndarray, labels: np.ndarray) -> tuple[bool, str]:
    """
    Verifica se a imagem e as labels têm dimensões espaciais compatíveis.

    Returns:
        (True, "") se compatível
        (False, mensagem_de_erro) se incompatível
    """
    img_h, img_w = image.shape[1], image.shape[2]
    lbl_h, lbl_w = labels.shape[0], labels.shape[1]
    if img_h != lbl_h or img_w != lbl_w:
        return False, (
            f"Imagem: {img_h}×{img_w} px | Labels: {lbl_h}×{lbl_w} px. "
            "As dimensões devem ser idênticas."
        )
    return True, ""


def get_band_stats(image: np.ndarray) -> pd.DataFrame:
    """
    Gera tabela de estatísticas básicas por banda.

    Args:
        image: Array (bands, height, width)

    Returns:
        DataFrame com min, max, média, desvio padrão e % de NaN por banda
    """
    rows = []
    for i in range(image.shape[0]):
        band = image[i].astype(np.float32)
        nan_pct = float(np.isnan(band).sum()) / band.size * 100
        rows.append({
            "Banda": f"Banda {i}",
            "Mín": round(float(np.nanmin(band)), 4),
            "Máx": round(float(np.nanmax(band)), 4),
            "Média": round(float(np.nanmean(band)), 4),
            "Desvio Padrão": round(float(np.nanstd(band)), 4),
            "% NaN": round(nan_pct, 2),
        })
    return pd.DataFrame(rows)


def get_label_stats(labels: np.ndarray) -> pd.DataFrame:
    """
    Gera tabela com contagem e percentual de pixels por classe.

    Args:
        labels: Array 2D de labels

    Returns:
        DataFrame com classe, contagem e percentual
    """
    valid = labels[labels > 0]
    total = len(valid)
    classes, counts = np.unique(valid, return_counts=True)
    rows = [
        {
            "Classe": int(c),
            "Pixels": int(n),
            "% do Total": round(100 * n / total, 2) if total > 0 else 0,
        }
        for c, n in zip(classes, counts)
    ]
    return pd.DataFrame(rows)


def format_metrics(metrics: dict) -> pd.DataFrame:
    """
    Formata as métricas principais do modelo em um DataFrame legível.

    Args:
        metrics: Dicionário retornado por train_model()

    Returns:
        DataFrame com métrica e valor
    """
    rows = [
        {"Métrica": "Acurácia (Treino)", "Valor": metrics.get("train_accuracy", "—")},
        {"Métrica": "OOB Score", "Valor": metrics.get("oob_score", "—")},
        {"Métrica": "Classes", "Valor": metrics.get("n_classes", "—")},
        {"Métrica": "Amostras de Treino", "Valor": metrics.get("n_samples", "—")},
        {"Métrica": "Features Usadas", "Valor": metrics.get("n_features", "—")},
    ]
    if "cv_mean" in metrics:
        rows.append({"Métrica": "CV Acurácia (média)", "Valor": metrics["cv_mean"]})
        rows.append({"Métrica": "CV Acurácia (std)", "Valor": metrics["cv_std"]})
    return pd.DataFrame(rows)


def fig_to_bytes(fig: plt.Figure) -> bytes:
    """
    Converte figura matplotlib em bytes PNG para download no Streamlit.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()