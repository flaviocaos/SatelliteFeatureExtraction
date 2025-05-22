
import matplotlib.pyplot as plt

def plot_classification(classified_map, output_path):
    plt.figure(figsize=(10, 6))
    plt.imshow(classified_map, cmap='jet')
    plt.title('Mapa Classificado')
    plt.colorbar(label='Classes')
    plt.axis('off')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
