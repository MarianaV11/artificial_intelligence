import numpy as np
from loguru import logger
import math


class BayesUnivariado:
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray):
        """
        Implementação do Classificador Bayesiano Univariado.
        Cada atributo é tratado como uma distribuição normal independente.

        :param x_train: matriz de treino (amostras x atributos)
        :param y_train: vetor de rótulos de treino
        """
        self.x_train = x_train
        self.y_train = y_train
        self.classes = np.unique(y_train)

        # Estatísticas da distribuição normal por classe e atributo
        self.mean = {}
        self.std = {}
        self.prior = {}

        logger.info("[BayesUnivariado] Iniciando treinamento...")
        self._fit()

    def _fit(self):
        """
        Calcula média, desvio padrão e probabilidade a priori (P(classe)).
        """
        for c in self.classes:
            x_class = self.x_train[self.y_train == c]
            self.mean[c] = np.mean(x_class, axis=0)
            self.std[c] = np.std(x_class, axis=0, ddof=1)
            self.prior[c] = len(x_class) / len(self.x_train)

        logger.info("[BayesUnivariado] Treinamento concluído.")

    def _gaussian_pdf(self, x, mean, std):
        """
        Calcula a densidade da normal univariada:
        p(x) = (1 / sqrt(2πσ²)) * exp(-0.5 * ((x - μ)/σ)²)
        """
        eps = 1e-9  # evitar divisão por zero
        exponent = np.exp(-((x - mean) ** 2) / (2 * (std + eps) ** 2))
        return (1 / (np.sqrt(2 * math.pi) * (std + eps))) * exponent

    def predict(self, x_test: np.ndarray):
        """
        Prediz as classes do conjunto de teste com base nas probabilidades máximas.
        """
        logger.info("[BayesUnivariado] Iniciando predição...")
        y_pred = []

        for x in x_test:
            posteriors = {}

            for c in self.classes:
                # P(x|c)
                likelihood = np.prod(self._gaussian_pdf(x, self.mean[c], self.std[c]))
                # P(c)
                prior = self.prior[c]
                # P(c|x) ~ P(x|c) * P(c)
                posteriors[c] = likelihood * prior

            # Classe com maior probabilidade posterior
            y_pred.append(max(posteriors, key=posteriors.get))

        logger.info("[BayesUnivariado] Predição concluída.")
        return np.array(y_pred)
