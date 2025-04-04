import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_origin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.preprocessing import load_image, normalize_image, calculate_ndvi
from src.export import export_raster

def main():
    image_path = "data/sentinel2_example.tif"

    print("Lendo e pré-processando a imagem...")
    image, profile = load_image(image_path)
    image_normalized = normalize_image(image)

    # Verifica se há bandas suficientes para NDVI (mínimo: NIR e Red)
    if image.shape[0] >= 5:
        ndvi = calculate_ndvi(image_normalized[4], image_normalized[3])
    else:
        ndvi = None
        print("NDVI não pôde ser calculado: número insuficiente de bandas.")

    print("Preparando dados para treinamento...")
    features = image_normalized.reshape(-1, image_normalized.shape[0])
    labels = np.random.randint(0, 2, size=(features.shape[0],))  # Substituir por rótulos reais
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    print("Treinando modelo Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print("Acurácia Random Forest:", accuracy_score(y_test, y_pred))

    print("Classificando toda a imagem...")
    classified = rf.predict(features).reshape(image.shape[1], image.shape[2])

    print("Exportando resultado...")
    export_raster("outputs/resultado_classificacao.tif", classified, profile)

    print("Visualizando resultado...")
    plt.figure(figsize=(10, 6))
    plt.imshow(classified, cmap='jet')
    plt.title('Mapa Classificado')
    plt.colorbar()
    plt.savefig("outputs/mapa_classificado.pdf", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
