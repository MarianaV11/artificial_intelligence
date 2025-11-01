import math

import numpy as np
from loguru import logger


class HoldOut:
    def __init__(self, x, y, test_size=0.2, random_state=42):
        self.x = x
        self.y = y

        self.test_size = test_size
        self.random_state = random_state


    def execute_method(self):
        np.random.seed(self.random_state)

        number_of_samples = len(self.x)
        indices = np.random.permutation(number_of_samples)
        logger.info(f"[HoldOut] Random indices: {indices}")

        number_of_tests = math.ceil(number_of_samples * self.test_size)

        test_indices = indices[:number_of_tests]
        train_indices = indices[number_of_tests:]

        if self.x.ndim == 1:
            x_train, x_test = self.x[train_indices], self.x[test_indices]
        else:
            x_train, x_test = self.x[train_indices, :], self.x[test_indices, :]

        y_train, y_test = self.y[train_indices], self.y[test_indices]

        return x_train, x_test, y_train, y_test
