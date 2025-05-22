# 🛰️ Satellite Feature Extraction

Projeto de classificação de uso e cobertura da terra utilizando imagens Sentinel-2 e aprendizado de máquina (Random Forest).

## 📁 Estrutura do Projeto

Lancover_classification/
  data/                  - Dados de entrada (imagens e labels)
  outputs/               - Saídas (rasters e figuras)
    rasters/             - Arquivos raster classificados
    figures/             - Mapas e gráficos exportados
  src/                   - Scripts e módulos Python
    preprocessing.py     - Processamento e normalização de imagens
    model.py             - Modelagem e classificação
    export.py            - Exportação de arquivos
    visualization.py     - Geração de mapas e gráficos
  LandCover_Classification.ipynb - Notebook principal do projeto
  main.py                - Pipeline completo via script
  .gitignore             - Arquivos ignorados pelo Git
  git_auto_commit.bat    - Script de automação de commits
  checklist_git.pdf      - Checklist dos comandos Git
  README.md              - Documentação do projeto
  requirements.txt       - Dependências necessárias para execução

## 🚀 Como Executar

### ✔️ Executar pelo pipeline (main.py)

1. Instale as dependências:

pip install -r requirements.txt

2. Execute o pipeline:

python main.py

### ✔️ Executar pelo notebook

1. Abra o arquivo LandCover_Classification.ipynb no Jupyter Notebook, Jupyter Lab ou VS Code.
2. Execute célula por célula.

## 🧠 Funcionalidades

- ✅ Carregamento de imagens Sentinel-2.
- ✅ Cálculo do NDVI.
- ✅ Pré-processamento das bandas espectrais.
- ✅ Treinamento de modelo Random Forest.
- ✅ Classificação supervisionada da imagem.
- ✅ Exportação dos resultados (raster e gráficos).
- ✅ Geração de mapas classificados.

## 🔧 Dependências

- numpy
- matplotlib
- scikit-learn
- rasterio

Instale todas com:

pip install -r requirements.txt

## 📄 Licença

Este projeto está licenciado sob a licença MIT. Consulte o arquivo LICENSE para mais informações.

## 🙌 Autor

**Flávio Caos**  
🔗 https://github.com/flaviocaos

## 🌟 Checklist Git incluído

- Arquivo checklist_git.pdf disponível no projeto.
- Script git_auto_commit.bat para automação de commits e pushes no Windows.

## 🚀 Fluxo Git Recomendo

- Branch main → produção
- Branch develop → desenvolvimento
- Branches feature/* → novas funcionalidades
- Branch hotfix/* → correções rápidas
- Branch release/* → preparação de releases