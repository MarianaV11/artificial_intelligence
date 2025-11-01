import numpy as np


class KNN:
    def __init__(self, x_train, y_train, k=5, task="C"):
        self.k = k
        self.task = task

        self.x_train = x_train
        self.y_train = y_train


    def predict(self, x_test):
        predictions_euclidean = [self.calculate_prediction(x) for x in x_test]
        
        predictions_manhatthan = [self.calculate_prediction(x, "manhatthan") for x in x_test]

        return np.array(predictions_euclidean), np.array(predictions_manhatthan)


    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum(x1 - x2) ** 2)


    def manhatthan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2))


    def calculate_prediction(self, x, distance_type="euclidean"):
        distances = []
        if distance_type == "euclidean":
            distances = [self.euclidean_distance(x, x_train) for x_train in self.x_train]
        else:
            distances = [self.manhatthan_distance(x, x_train) for x_train in self.x_train]

        k_indices = np.argsort(distances)[: self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        if self.task == "C":
            unique, counts = np.unique(k_nearest_labels, return_counts=True)

            return unique[np.argmax(counts)]
        elif self.task == "R":
            return np.mean(k_nearest_labels)
        else:
            raise ValueError("Tarefa nao foi definida")
