import numpy as np
from loguru import logger
import math


class BayesMultivariado:
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray):
        self.x_train = x_train
        self.y_train = y_train
        self.classes = np.unique(y_train)

        self.mean = {}
        self.cov = {}
        self.prior = {}

        logger.info("[BayesMultivariado] Iniciando treinamento...")
        self._fit()

    def _fit(self):
        for c in self.classes:
            x_class = self.x_train[self.y_train == c]
            self.mean[c] = np.mean(x_class, axis=0)
            self.cov[c] = np.cov(x_class, rowvar=False)
            self.prior[c] = len(x_class) / len(self.x_train)

        logger.info("[BayesMultivariado] Treinamento concluído.")

    def _multivariate_gaussian_pdf(self, x, mean, cov):
        d = len(mean)
        eps = 1e-9  # estabilidade numérica
        cov_det = np.linalg.det(cov + np.eye(d) * eps)
        cov_inv = np.linalg.inv(cov + np.eye(d) * eps)
        x_mu = x - mean
        exponent = -0.5 * np.dot(np.dot(x_mu.T, cov_inv), x_mu)
        return (1 / ((2 * math.pi) ** (d / 2) * (cov_det ** 0.5))) * np.exp(exponent)

    def predict(self, x_test: np.ndarray):
        logger.info("[BayesMultivariado] Iniciando predição...")
        y_pred = []

        for x in x_test:
            posteriors = {}

            for c in self.classes:
                likelihood = self._multivariate_gaussian_pdf(x, self.mean[c], self.cov[c])
                prior = self.prior[c]
                posteriors[c] = likelihood * prior

            y_pred.append(max(posteriors, key=posteriors.get))

        logger.info("[BayesMultivariado] Predição concluída.")
        return np.array(y_pred)
