import matplotlib.pyplot as plt
import numpy as np
from loguru import logger


class Metrics:
    def confusion_matrix(self, y_predict, y_real, name, graph_color):
        classes = np.unique(y_real)
        num_classes = len(classes)

        conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

        # Preenche a matriz de confusão
        for i, true_label in enumerate(classes):
            for j, pred_label in enumerate(classes):
                conf_matrix[i, j] = np.sum(
                    (y_real == true_label) & (y_predict == pred_label)
                )

        logger.info(f"Confusion Matrix:\n{conf_matrix}")

        # plt.imshow(conf_matrix, cmap=graph_color)
        # plt.title("Matriz de Confusão")
        # plt.colorbar()

        # # Ajuste correto dos ticks
        # plt.xticks(range(num_classes), classes)
        # plt.yticks(range(num_classes), classes)

        # plt.xlabel("Predito")
        # plt.ylabel("Verdadeiro")

        # # Inserir valores dentro dos quadrados
        # for i in range(num_classes):
        #     for j in range(num_classes):
        #         plt.text(
        #             j,
        #             i,
        #             str(conf_matrix[i, j]),
        #             ha="center",
        #             va="center",
        #             color="black",
        #             fontsize=12,
        #         )

        # plt.savefig(f"assets/{name}.png", dpi=300, bbox_inches="tight")
        # plt.close()

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
        total = conf_matrix.sum()
        correct = np.trace(conf_matrix)

        if total == 0:
            return 0.0

        accuracy_val = correct / total
        logger.info(f"Accuracy: {accuracy_val:.4f}")
        return accuracy_val

    def precision(self, conf_matrix):
        precisions = []

        for i in range(conf_matrix.shape[0]):
            TP = conf_matrix[i, i]
            FP = conf_matrix[:, i].sum() - TP

            if TP + FP == 0:
                precisions.append(0)
            else:
                precisions.append(TP / (TP + FP))

        macro_precision = np.mean(precisions)
        logger.info(f"Precision (macro): {macro_precision:.4f}")
        return macro_precision


    def recall(self, conf_matrix):
        recalls = []

        for i in range(conf_matrix.shape[0]):
            TP = conf_matrix[i, i]
            FN = conf_matrix[i, :].sum() - TP

            if TP + FN == 0:
                recalls.append(0)
            else:
                recalls.append(TP / (TP + FN))

        macro_recall = np.mean(recalls)
        logger.info(f"Recall (macro): {macro_recall:.4f}")
        return macro_recall


    def f1_score(self, conf_matrix):
        precision_val = self.precision(conf_matrix)
        recall_val = self.recall(conf_matrix)

        if precision_val + recall_val == 0:
            return 0.0

        f1 = 2 * (precision_val * recall_val) / (precision_val + recall_val)
        logger.info(f"F1-score (macro): {f1:.4f}")
        return f1

