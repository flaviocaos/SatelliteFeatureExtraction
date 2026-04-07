# core/visualization.py
"""
Módulo de visualização: mapas, bandas, NDVI e importância de features.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure


# Paleta de cores discreta para até 10 classes de uso da terra
LAND_COVER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
]


def plot_classification(
    classified_map: np.ndarray,
    output_path: str | None = None,
    class_names: dict | None = None,
) -> Figure:
    """
    Gera mapa classificado com paleta discreta e legenda.

    Args:
        classified_map: Array 2D com classes inteiras
        output_path: Se informado, salva o arquivo
        class_names: Dicionário {int_class: "nome"} para legenda customizada

    Returns:
        fig: Figura matplotlib
    """
    classes = np.unique(classified_map[classified_map > 0])
    n_classes = len(classes)

    cmap = mcolors.ListedColormap(LAND_COVER_COLORS[:n_classes])
    bounds = np.arange(classes.min(), classes.max() + 2)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(classified_map, cmap=cmap, norm=norm)
    ax.set_title("Mapa de Classificação de Uso e Cobertura da Terra", fontsize=13, pad=12)
    ax.axis("off")

    cbar = fig.colorbar(im, ax=ax, ticks=classes, fraction=0.03, pad=0.04)
    if class_names:
        cbar.set_ticklabels([class_names.get(c, f"Classe {c}") for c in classes])
    else:
        cbar.set_ticklabels([f"Classe {c}" for c in classes])
    cbar.set_label("Classes", fontsize=10)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_band(
    band: np.ndarray,
    title: str = "Banda",
    cmap: str = "gray",
) -> Figure:
    """
    Visualiza uma única banda espectral com estiramento percentil (2–98%).

    Args:
        band: Array 2D da banda
        title: Título do gráfico
        cmap: Colormap matplotlib

    Returns:
        fig: Figura matplotlib
    """
    p2, p98 = np.nanpercentile(band, [2, 98])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(band, cmap=cmap, vmin=p2, vmax=p98)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    return fig


def plot_ndvi(ndvi: np.ndarray, output_path: str | None = None) -> Figure:
    """
    Visualiza o NDVI com colormap RdYlGn.

    Args:
        ndvi: Array 2D com valores em [-1, 1]
        output_path: Se informado, salva o arquivo

    Returns:
        fig: Figura matplotlib
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_title("NDVI — Índice de Vegetação por Diferença Normalizada", fontsize=11, pad=10)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="NDVI")
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_feature_importance(
    importance_dict: dict,
    title: str = "Importância das Features",
) -> Figure:
    """
    Gráfico de barras horizontais com importância das features.

    Args:
        importance_dict: {feature_name: importância} ordenado
        title: Título do gráfico

    Returns:
        fig: Figura matplotlib
    """
    names = list(importance_dict.keys())
    values = list(importance_dict.values())

    fig, ax = plt.subplots(figsize=(7, max(3, len(names) * 0.4)))
    bars = ax.barh(names, values, color="#2ca02c", edgecolor="white")
    ax.set_xlabel("Importância", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.invert_yaxis()
    for bar, val in zip(bars, values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray, classes: list) -> Figure:
    """
    Matriz de confusão normalizada.

    Args:
        cm: Matriz de confusão (sklearn)
        classes: Lista de nomes das classes

    Returns:
        fig: Figura matplotlib
    """
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(max(4, len(classes)), max(3, len(classes))))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão (Normalizada)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                    fontsize=8)
    plt.tight_layout()
    return fig
