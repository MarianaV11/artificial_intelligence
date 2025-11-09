import numpy as np
from loguru import logger

class KFoldCV:
    def __init__(self, x, y, n_splits=10, random_state=42):
        self.x = x
        self.y = y
        self.n_splits = n_splits
        self.random_state = random_state

    def execute_method(self):
        np.random.seed(self.random_state)
        number_of_samples = len(self.x)

        # Embaralha os índices
        indices = np.random.permutation(number_of_samples)
        logger.info(f"[KFoldCV] Random indices: {indices}")

        # Divide os índices em n_splits partes (folds)
        folds = np.array_split(indices, self.n_splits)
        logger.info(f"[KFoldCV] Fold sizes: {[len(f) for f in folds]}")

        # Gera os conjuntos de treino e teste para cada fold
        results = []
        for i in range(self.n_splits):
            test_indices = folds[i]
            train_indices = np.hstack([folds[j] for j in range(self.n_splits) if j != i])

            if self.x.ndim == 1:
                x_train, x_test = self.x[train_indices], self.x[test_indices]
            else:
                x_train, x_test = self.x[train_indices, :], self.x[test_indices, :]

            y_train, y_test = self.y[train_indices], self.y[test_indices]

            results.append((x_train, x_test, y_train, y_test))
            logger.info(f"[KFoldCV] Fold {i+1}/{self.n_splits} - Train: {len(train_indices)}, Test: {len(test_indices)}")

        return results
