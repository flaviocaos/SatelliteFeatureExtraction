# 🛰️ Satellite Feature Extraction

Classificação de **Uso e Cobertura da Terra** com imagens **Sentinel-2** e **Random Forest**.  
Projeto de TCC com interface web local construída em Streamlit.

**Autor:** Flávio Caos · [github.com/flaviocaos](https://github.com/flaviocaos)

---

## 📁 Estrutura do Projeto

```
SatelliteFeatureExtraction/
├── app.py                  # Ponto de entrada — streamlit run app.py
├── requirements.txt
├── README.md
├── data/                   # Imagens de entrada (não versionado)
├── outputs/
│   ├── rasters/            # Rasters classificados exportados
│   └── figures/            # Figuras e mapas exportados
├── core/
│   ├── __init__.py
│   ├── preprocessing.py    # load_image, normalize, NDVI, load_labels
│   ├── model.py            # train_model, classify_image, feature_importance
│   ├── export.py           # export_raster, export_figure
│   └── visualization.py    # plot_classification, plot_band, plot_ndvi, etc.
├── utils/
│   ├── __init__.py
│   └── helpers.py          # validate_shapes, get_band_stats, format_metrics
├── assets/                 # Logos e recursos estáticos
└── LandCover_Classification.ipynb
```

---

## 🚀 Como Executar

### 1. Criar ambiente virtual

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Rodar o app

```bash
streamlit run app.py
```

O app abrirá automaticamente em `http://localhost:8501`.

---

## 🧠 Pipeline do App

| Etapa | Descrição |
|---|---|
| **1. Upload** | Imagem Sentinel-2 multibanda + raster de labels |
| **2. Exploração** | Estatísticas por banda, visualização e análise de classes |
| **3. Pré-processamento** | Seleção de bandas, normalização min-max, cálculo de NDVI |
| **4. Treinamento** | Random Forest com OOB score, cross-validation e importância de features |
| **5. Classificação** | Predição pixel a pixel com o modelo treinado |
| **6. Exportação** | Raster GeoTIFF comprimido (LZW) + figuras PNG |

---

## ⚙️ Parâmetros Configuráveis (Sidebar)

- Índices das bandas NIR e Red para o NDVI
- `n_estimators`, `max_depth`, `random_state`
- Número de folds para cross-validation
- Ativar/desativar NDVI como feature

---

## 📦 Dependências Principais

| Biblioteca | Uso |
|---|---|
| `streamlit` | Interface web |
| `rasterio` | I/O de rasters GeoTIFF |
| `scikit-learn` | Random Forest e métricas |
| `numpy` | Manipulação de arrays |
| `matplotlib` | Visualizações e mapas |
| `pandas` | Tabelas de estatísticas |

---

## 🔮 Melhorias Futuras

- [ ] Suporte a CNNs (U-Net, ResNet) para segmentação semântica
- [ ] Múltiplos classificadores (SVM, XGBoost, KNN) com comparação
- [ ] Geração automática de relatório PDF com resultados
- [ ] Suporte a mosaicos grandes com processamento em tiles
- [ ] Exibição de métricas por classe (F1-score, recall, precision)
- [ ] Integração com Google Earth Engine para download de imagens
- [ ] Versão Docker para deploy simplificado
