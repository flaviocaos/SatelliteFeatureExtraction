# Extração de Feições em Imagens de Satélite com Machine Learning

Este projeto tem como objetivo desenvolver um pipeline em Python para extrair feições de imagens de satélite de baixa, média e alta resolução espacial utilizando algoritmos de aprendizado de máquina como Random Forest e redes neurais convolucionais (CNNs).

## 📂 Estrutura do Projeto

```
satellite-feature-extraction/
├── data/                    # Imagens de entrada (GeoTIFF)
├── outputs/                 # Resultados exportados (GeoTIFF, PDF)
├── src/                     # Código fonte
│   ├── preprocessing.py     # Leitura e normalização de imagens, NDVI
│   └── export.py            # Função para exportar o raster classificado
├── main.py                  # Script principal
├── requirements.txt         # Bibliotecas necessárias
└── README.md
```

## ⚙️ Requisitos

- Python 3.9+
- Bibliotecas:
  - numpy
  - matplotlib
  - scikit-learn
  - rasterio
  - tensorflow

Instale com:

```bash
pip install -r requirements.txt
```

## 🚀 Como Executar

1. Coloque a imagem de satélite desejada em `data/` (ex: Sentinel-2, MODIS, WorldView).
2. Edite o caminho da imagem no arquivo `main.py`:

```python
image_path = "data/sentinel2_example.tif"
```

3. Execute o script principal:

```bash
python main.py
```

4. O resultado será salvo em:
- GeoTIFF: `outputs/resultado_classificacao.tif`
- Mapa em PDF: `outputs/mapa_classificado.pdf`

## 📌 Observações

- O projeto está preparado para lidar com diferentes resoluções espaciais (baixa, média e alta).
- Atualmente os rótulos são gerados aleatoriamente para fins de teste.
- Para uso real, recomenda-se substituir por dados de treinamento reais (shapefiles, máscaras raster ou amostras manuais).

## 📈 Extensões Futuras
- Integração com shapefiles para rótulos supervisionados
- Suporte a CNNs com extração de patches
- Automatização de recorte, reprojeção e amostragem

---

Desenvolvido como parte de um TCC sobre Machine Learning e Sensoriamento Remoto 🌍📡

