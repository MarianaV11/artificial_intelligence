import matplotlib.pyplot as plt
import numpy as np
from loguru import logger


class Metrics:
    def confusion_matrix(self, y_predict, y_real, name, graph_color):
        classes = np.unique(y_real)

        conf_matrix = np.zeros((len(classes), len(classes)), dtype=int)

        for i, true_label in enumerate(classes):
            for j, pred_label in enumerate(classes):
                conf_matrix[i, j] = np.sum(
                    (y_real == true_label) & (y_predict == pred_label)
                )

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
                plt.text(
                    j,
                    i,
                    str(conf_matrix[i, j]),
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=12,
                )

        plt.savefig(f"assets/{name}.png", dpi=300, bbox_inches="tight")
        plt.close()

        return conf_matrix

    def get_metrics_components(self, conf_matrix):
        if conf_matrix.shape != (2, 2):
            logger.error("The confusion matrix must be 2x2 for these metrics.")
            return 0, 0, 0, 0

        TN = conf_matrix[0, 0]
        FP = conf_matrix[0, 1]
        FN = conf_matrix[1, 0]
        TP = conf_matrix[1, 1]

        return TN, FP, FN, TP

    def accuracy(self, conf_matrix):
        TN, FP, FN, TP = self.get_metrics_components(conf_matrix)

        total = TN + FP + FN + TP
        if total == 0:
            return 0.0

        accuracy_val = (TP + TN) / total
        logger.info(f"Accuracy: {accuracy_val:.4f}")

        return accuracy_val

    def precision(self, conf_matrix):
        TN, FP, FN, TP = self.get_metrics_components(conf_matrix)

        divisor = TP + FP
        if divisor == 0:
            return 0.0

        precision_val = TP / divisor
        logger.info(f"Precision: {precision_val:.4f}")

        return precision_val

    def f1_score(self, conf_matrix):
        precision_val = self.precision(conf_matrix)
        recall_val = self.recall(conf_matrix)

        divisor = precision_val + recall_val
        if divisor == 0:
            return 0.0

        f1_val = 2 * (precision_val * recall_val) / divisor
        logger.info(f"F1-Score: {f1_val:.4f}")

        return f1_val

    def recall(self, conf_matrix):
        TN, FP, FN, TP = self.get_metrics_components(conf_matrix)

        divisor = TP + FN
        if divisor == 0:
            return 0.0

        recall_val = TP / divisor
        logger.info(f"Recall: {recall_val:.4f}")
        return recall_val
