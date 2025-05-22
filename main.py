
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.preprocessing import load_image, normalize_image, calculate_ndvi, load_labels
from src.model import train_model, classify_image
from src.export import export_raster
from src.visualization import plot_classification

def main():
    image_path = "data/sentinel2_example.tif"
    label_path = "data/labels.tif"
    output_raster = "outputs/rasters/resultado_classificacao.tif"
    output_figure = "outputs/figures/mapa_classificado.pdf"

    print("🚀 Lendo imagem...")
    image, profile = load_image(image_path)
    image_normalized = normalize_image(image)

    print("🧠 Calculando NDVI...")
    if image.shape[0] >= 5:
        ndvi = calculate_ndvi(image_normalized[4], image_normalized[3])
        image_with_ndvi = np.vstack([image_normalized, ndvi[None, ...]])
    else:
        print("⚠️ NDVI não pôde ser calculado. Prosseguindo sem ele.")
        image_with_ndvi = image_normalized

    print("🏷️ Lendo rótulos...")
    labels, _ = load_labels(label_path)
    labels_flat = labels.flatten()
    mask = labels_flat != 0

    features = image_with_ndvi.reshape(image_with_ndvi.shape[0], -1).T
    features = features[mask]
    labels_flat = labels_flat[mask]

    print("📊 Separando treino e teste...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels_flat, test_size=0.3, random_state=42
    )

    print("🎯 Treinando modelo...")
    model = train_model(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"✅ Acurácia: {accuracy_score(y_test, y_pred) * 100:.2f}%")

    print("🗺️ Classificando imagem completa...")
    classified = classify_image(model, image_with_ndvi)

    print("💾 Exportando raster...")
    export_raster(output_raster, classified, profile)

    print("🎨 Gerando mapa...")
    plot_classification(classified, output_figure)

    print("🏁 Pipeline finalizado com sucesso!")

if __name__ == "__main__":
    main()
