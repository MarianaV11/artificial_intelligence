import numpy as np

class MLP:
    def __init__(self, input_dim, hidden=[10, 10], lr=0.1, epochs=300):
        self.lr = lr
        self.epochs = epochs
        self.W = []
        self.loss_history = []
        self.init_weights(input_dim, hidden)

    def init_weights(self, input_dim, hidden):
        layers = [input_dim] + hidden + [1]
        for i in range(len(layers) - 1):
            self.W.append(np.random.uniform(-0.5, 0.5, (layers[i + 1], layers[i] + 1)))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return x * (1 - x)

    def forward(self, x):
        self.y = []
        for w in self.W:
            x = np.insert(x, 0, -1)
            x = self.sigmoid(np.dot(w, x))
            self.y.append(x)
        return self.y[-1]

    def backward(self, x, d):
        deltas = [None] * len(self.W)
        deltas[-1] = self.sigmoid_deriv(self.y[-1]) * (d - self.y[-1])
        for l in reversed(range(len(self.W) - 1)):
            W_no_bias = self.W[l + 1][:, 1:]
            deltas[l] = self.sigmoid_deriv(self.y[l]) * np.dot(
                W_no_bias.T, deltas[l + 1]
            )

        input_vals = []
        x_bias = np.insert(x, 0, -1)
        input_vals.append(x_bias)
        for out in self.y[:-1]:
            input_vals.append(np.insert(out, 0, -1))

        for l in range(len(self.W)):
            self.W[l] += self.lr * np.outer(deltas[l], input_vals[l])

    def fit(self, X, Y):
        N = X.shape[0]
        for _ in range(self.epochs):
            eqm = 0
            for x, d in zip(X, Y):
                y = self.forward(x)
                self.backward(x, d)
                eqm += np.mean((d - y) ** 2)
            self.loss_history.append(eqm / N)

    def predict(self, X):
        return np.array([self.forward(x) for x in X]).squeeze()