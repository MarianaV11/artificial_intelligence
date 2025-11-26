import numpy as np


class Perceptron:
    def __init__(self, x, y,learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        
        self.x = x
        self.y = y
        
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self):
        n_samples, n_features = self.x.shape

        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.epochs):
            for idx, x_i in enumerate(self.x):
                linear_output = np.dot(x_i, self.weights)+self.bias
                y_predicted = self._step_function(linear_output)
                if self.y[idx] != y_predicted:
                    update = self.learning_rate * (self.y[idx] - y_predicted)
                    self.weights += update *x_i
                    self.bias +=update

    def _step_function(self,x):
        return np.where(x>=0, 1, 0)

    def predict(self, x_test):
        # Se não treinou, inicializa valores neutros
        if self.weights is None:
            self.weights = np.zeros(x_test.shape[1])

        if self.bias is None:
            self.bias = 0.0

        linear_output = np.dot(x_test, self.weights) + self.bias
        y_predicted = self._step_function(linear_output)
        return y_predicted
