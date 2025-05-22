# 🛰️ LandCover Classification

Este projeto realiza a **classificação de uso e cobertura da terra** com base em imagens multiespectrais do Sentinel-2, utilizando processamento com Python e aprendizado de máquina com Random Forest.

---

## 📁 Estrutura do Projeto

```
Lancover_classification/
├── data/                       # Dados de entrada (.tif)
├── outputs/                    # Resultados (raster e figuras)
├── src/                        # Módulos Python (preprocessamento, modelo, exportação, visualização)
├── LandCover_Classification.ipynb   # ✅ Notebook interativo
├── main.py                     # Pipeline automatizado via script
├── requirements.txt             # Dependências
├── .gitignore                   # Arquivos ignorados no Git
└── README.md                    # Documentação
```

---

## ▶️ Como executar o pipeline (script)

1. Garanta que todas as dependências estejam instaladas:

```bash
pip install -r requirements.txt
```

2. Execute o pipeline:

```bash
python main.py
```

---

## 📓 Como usar o notebook

1. Abra o `LandCover_Classification.ipynb` no **Jupyter Notebook**, **Jupyter Lab**, **VS Code (com extensão Jupyter)** ou **Google Colab**.
2. Execute célula por célula para visualizar todo o processo, desde a leitura das imagens até a geração dos mapas classificados.

---

## 🧠 Funcionalidades

- Carregamento de imagens satélite Sentinel-2.
- Cálculo do NDVI.
- Pré-processamento de dados raster.
- Treinamento de modelo Random Forest.
- Classificação supervisionada da imagem.
- Exportação dos resultados em raster e em figuras.
- Visualização dos mapas classificados.

---

## 🧪 Tecnologias utilizadas

- Python 3
- NumPy
- Scikit-learn
- Matplotlib
- Rasterio
- Jupyter Notebook

---

## 🚀 Melhorias futuras

- Integração com outros índices espectrais (NDWI, NDBI).
- Interface gráfica para uso não programático.
- Integração com plataformas em nuvem (Google Earth Engine, AWS).
- Aumento na diversidade de datasets.

---

## 📝 Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

## 👨‍💻 Autor

**Flavio Caos**  
🔗 [https://github.com/flaviocaos](https://github.com/flaviocaos)