# app.py
"""
Satellite Feature Extraction — Interface Streamlit
Execução: streamlit run app.py
"""

import os
import tempfile

import numpy as np
import pandas as pd
import streamlit as st

from core.preprocessing import load_image, normalize_image, calculate_ndvi, load_labels
from core.model import train_model, classify_image, get_feature_importance
from core.export import export_raster, export_figure
from core.visualization import (
    plot_band,
    plot_ndvi,
    plot_classification,
    plot_feature_importance,
    plot_confusion_matrix,
)
from utils.helpers import (
    validate_shapes,
    get_band_stats,
    get_label_stats,
    format_metrics,
    fig_to_bytes,
)

# ── Configuração da página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Satellite Feature Extraction",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estado da sessão ─────────────────────────────────────────────────────────
KEYS = [
    "image", "profile", "labels", "label_profile",
    "normalized", "ndvi", "model", "metrics",
    "classified", "selected_bands", "feature_names",
    "tmp_image", "tmp_labels",
]
for k in KEYS:
    if k not in st.session_state:
        st.session_state[k] = None

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://sentinel.esa.int/documents/247904/266422/Sentinel-2.jpg", width='stretch')
    st.title("🛰️ Configurações")

    st.subheader("Bandas Espectrais")
    nir_idx = st.number_input("Índice banda NIR", min_value=0, value=3,
                               help="Índice 0-based da banda NIR (tipicamente B8 no Sentinel-2)")
    red_idx = st.number_input("Índice banda Red", min_value=0, value=2,
                               help="Índice 0-based da banda Red (tipicamente B4 no Sentinel-2)")

    st.subheader("Random Forest")
    n_estimators = st.slider("n_estimators", 10, 500, 100, step=10)
    max_depth = st.selectbox(
        "max_depth",
        options=[None, 5, 10, 20, 50],
        format_func=lambda x: "Sem limite" if x is None else str(x),
    )
    random_state = st.number_input("random_state", value=42, step=1)
    cv_folds = st.slider("Cross-validation folds (0 = desativar)", 0, 10, 5)

    st.subheader("Features")
    use_ndvi_feature = st.checkbox("Incluir NDVI como feature", value=True)

    st.divider()
    st.caption("Satellite Feature Extraction · TCC · Flávio Caos")

# ── Header principal ─────────────────────────────────────────────────────────
st.title("🛰️ Satellite Feature Extraction")
st.markdown(
    """
    Classificação de **Uso e Cobertura da Terra** com imagens **Sentinel-2** e **Random Forest**.
    Faça upload da imagem multibanda e do raster de labels para iniciar o pipeline.
    """
)

# ── SEÇÃO 1: Upload ──────────────────────────────────────────────────────────
st.header("1 · Upload de Dados")
col1, col2 = st.columns(2)

with col1:
    st.subheader("📁 Imagem Raster")
    up_img = st.file_uploader("Arquivo .tif multibanda (Sentinel-2)", type=["tif", "tiff"])
    if up_img:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
        tmp.write(up_img.read())
        tmp.close()
        st.session_state.tmp_image = tmp.name
        try:
            img, profile = load_image(tmp.name)
            st.session_state.image = img
            st.session_state.profile = profile
            st.success(
                f"✅ {img.shape[0]} bandas · {img.shape[1]}×{img.shape[2]} px · "
                f"CRS: {profile.get('crs', 'não definido')}"
            )
        except Exception as e:
            st.error(f"Erro ao carregar imagem: {e}")

with col2:
    st.subheader("🏷️ Labels Raster")
    up_lbl = st.file_uploader("Arquivo .tif de labels (opcional)", type=["tif", "tiff"])
    if up_lbl:
        tmp_l = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
        tmp_l.write(up_lbl.read())
        tmp_l.close()
        st.session_state.tmp_labels = tmp_l.name
        try:
            labels, lbl_profile = load_labels(tmp_l.name)
            st.session_state.labels = labels
            st.session_state.label_profile = lbl_profile
            classes = np.unique(labels[labels > 0])
            st.success(f"✅ {len(classes)} classes · valores: {classes.tolist()}")
        except Exception as e:
            st.error(f"Erro ao carregar labels: {e}")

# ── SEÇÃO 2: Exploração ──────────────────────────────────────────────────────
if st.session_state.image is not None:
    img = st.session_state.image
    n_bands = img.shape[0]

    st.divider()
    st.header("2 · Exploração da Imagem")

    tab_stats, tab_band, tab_labels = st.tabs(["📊 Estatísticas", "🖼️ Bandas", "🏷️ Labels"])

    with tab_stats:
        st.dataframe(get_band_stats(img), width='stretch')

    with tab_band:
        band_idx = st.slider("Selecionar banda", 0, n_bands - 1, 0)
        fig_b = plot_band(img[band_idx], title=f"Banda {band_idx}")
        st.pyplot(fig_b, width='stretch')

    with tab_labels:
        if st.session_state.labels is not None:
            st.dataframe(get_label_stats(st.session_state.labels), width='stretch')
        else:
            st.info("Nenhum arquivo de labels carregado.")

# ── SEÇÃO 3: Pré-processamento ───────────────────────────────────────────────
if st.session_state.image is not None:
    img = st.session_state.image
    n_bands = img.shape[0]

    st.divider()
    st.header("3 · Pré-processamento")

    col_pre1, col_pre2 = st.columns([2, 1])

    with col_pre1:
        selected = st.multiselect(
            "Bandas para usar como features",
            options=list(range(n_bands)),
            default=list(range(min(n_bands, 6))),
            format_func=lambda i: f"Banda {i}",
        )

    with col_pre2:
        st.write("")
        st.write("")
        if st.button("🔄 Normalizar bandas", width='stretch'):
            if not selected:
                st.warning("Selecione ao menos uma banda.")
            else:
                with st.spinner("Normalizando..."):
                    norm = normalize_image(img[selected])
                    st.session_state.normalized = norm
                    st.session_state.selected_bands = selected
                    st.success(f"✅ Shape normalizado: {norm.shape}")

    # NDVI
    col_ndvi1, col_ndvi2 = st.columns([1, 2])
    with col_ndvi1:
        st.write("")
        if st.button("🌿 Calcular NDVI", width='stretch'):
            if nir_idx >= n_bands or red_idx >= n_bands:
                st.error(f"Índices inválidos — imagem tem {n_bands} bandas.")
            else:
                ndvi = calculate_ndvi(img[nir_idx], img[red_idx])
                st.session_state.ndvi = ndvi
                st.success("NDVI calculado!")
    with col_ndvi2:
        if st.session_state.ndvi is not None:
            fig_ndvi = plot_ndvi(st.session_state.ndvi)
            st.pyplot(fig_ndvi, width='stretch')
            st.download_button(
                "⬇️ Baixar NDVI (PNG)",
                data=fig_to_bytes(fig_ndvi),
                file_name="ndvi.png",
                mime="image/png",
            )

# ── SEÇÃO 4: Treinamento ─────────────────────────────────────────────────────
ready_to_train = (
    st.session_state.normalized is not None
    and st.session_state.labels is not None
)

if st.session_state.normalized is not None:
    st.divider()
    st.header("4 · Treinamento do Modelo")

    if not ready_to_train:
        st.info("ℹ️ Carregue um raster de labels para habilitar o treinamento.")

    if ready_to_train:
        if st.button("🚂 Treinar Random Forest", use_container_width=False, type="primary"):
            norm = st.session_state.normalized
            labels = st.session_state.labels
            ok, msg = validate_shapes(norm, labels)
            if not ok:
                st.error(msg)
            else:
                with st.spinner("Treinando modelo…"):
                    try:
                        n_b, h, w = norm.shape
                        X = norm.reshape(n_b, -1).T

                        # Monta lista de nomes das features
                        feat_names = [f"Banda {i}" for i in st.session_state.selected_bands]

                        if use_ndvi_feature and st.session_state.ndvi is not None:
                            X = np.hstack([X, st.session_state.ndvi.reshape(-1, 1)])
                            feat_names.append("NDVI")

                        st.session_state.feature_names = feat_names

                        y = labels.reshape(-1)
                        mask = y > 0
                        X_tr, y_tr = X[mask], y[mask]

                        model, metrics = train_model(
                            X_tr, y_tr,
                            n_estimators=int(n_estimators),
                            max_depth=max_depth,
                            random_state=int(random_state),
                            cv_folds=int(cv_folds),
                        )
                        st.session_state.model = model
                        st.session_state.metrics = metrics
                        st.success("✅ Modelo treinado!")
                    except Exception as e:
                        st.error(f"Erro no treinamento: {e}")

    if st.session_state.metrics is not None:
        metrics = st.session_state.metrics
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.subheader("📈 Métricas")
            st.dataframe(format_metrics(metrics), width='stretch')

        with col_m2:
            st.subheader("🌲 Importância das Features")
            if st.session_state.feature_names:
                imp = get_feature_importance(
                    st.session_state.model, st.session_state.feature_names
                )
                fig_imp = plot_feature_importance(imp)
                st.pyplot(fig_imp, width='stretch')

        # Matriz de confusão
        if "confusion_matrix" in metrics:
            with st.expander("🔢 Matriz de Confusão"):
                classes_list = [
                    f"Cls {c}" for c in range(metrics["confusion_matrix"].shape[0])
                ]
                fig_cm = plot_confusion_matrix(metrics["confusion_matrix"], classes_list)
                st.pyplot(fig_cm, width='stretch')

# ── SEÇÃO 5: Classificação ───────────────────────────────────────────────────
if st.session_state.model is not None and st.session_state.normalized is not None:
    st.divider()
    st.header("5 · Classificação da Imagem")

    if st.button("🗺️ Classificar Imagem", type="primary"):
        with st.spinner("Classificando todos os pixels…"):
            try:
                norm = st.session_state.normalized
                n_b, h, w = norm.shape
                X_full = norm.reshape(n_b, -1).T

                if use_ndvi_feature and st.session_state.ndvi is not None:
                    X_full = np.hstack([X_full, st.session_state.ndvi.reshape(-1, 1)])

                classified = classify_image(st.session_state.model, X_full, h, w)
                st.session_state.classified = classified
                st.success("✅ Classificação concluída!")
            except Exception as e:
                st.error(f"Erro na classificação: {e}")

    if st.session_state.classified is not None:
        classified = st.session_state.classified
        fig_cls = plot_classification(classified)
        st.pyplot(fig_cls, width='stretch')

        # ── SEÇÃO 6: Exportação ──────────────────────────────────────────────
        st.divider()
        st.header("6 · Exportação")
        col_ex1, col_ex2 = st.columns(2)

        with col_ex1:
            st.subheader("📦 Raster Classificado (.tif)")
            if st.button("💾 Exportar Raster", width='stretch'):
                out_path = os.path.join("outputs", "rasters", "classified.tif")
                try:
                    export_raster(out_path, classified, st.session_state.profile)
                    st.success(f"Salvo em: `{out_path}`")
                except Exception as e:
                    st.error(f"Erro ao exportar raster: {e}")

        with col_ex2:
            st.subheader("🖼️ Mapa Classificado (.png)")
            st.download_button(
                label="⬇️ Baixar Mapa (PNG)",
                data=fig_to_bytes(fig_cls),
                file_name="classified_map.png",
                mime="image/png",
                width='stretch',
            )
            if st.session_state.ndvi is not None:
                fig_ndvi_exp = plot_ndvi(st.session_state.ndvi)
                st.download_button(
                    label="⬇️ Baixar NDVI (PNG)",
                    data=fig_to_bytes(fig_ndvi_exp),
                    file_name="ndvi_map.png",
                    mime="image/png",
                    width='stretch',
                )