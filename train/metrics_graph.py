import matplotlib.pyplot as plt
import numpy as np
from loguru import logger


class Metrics:
    @staticmethod
    def confusion_matrix(y_predict, y_real, name, graph_color):
        classes = np.unique(y_real)
        
        conf_matrix = np.zeros((len(classes), len(classes)), dtype=int)

        for i, true_label in enumerate(classes):
            for j, pred_label in enumerate(classes):
                conf_matrix[i, j] = np.sum((y_real == true_label) & (y_predict == pred_label))

        logger.info("Confusion Matrix:\n", conf_matrix)
        
        plt.imshow(conf_matrix, graph_color)
        plt.title("Matriz de Confusão")
        plt.colorbar()
        plt.xticks([0, 1], classes)
        plt.yticks([0, 1], classes)
        plt.xlabel("Predito")
        plt.ylabel("Verdadeiro")

        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='black', fontsize=12)

        plt.savefig(f"assets/{name}.png", dpi=300, bbox_inches='tight')
        plt.close()