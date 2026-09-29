# src/model.py
import numpy as np

class CustomBGDRegressor:
    def __init__(self, learning_rate=0.01, epochs=500):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0
        self.cost_history = []

    def fit(self, X, y):
        """Trains the model using Batch Gradient Descent and MSE loss."""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.cost_history = []

        for epoch in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias

            # Gradients of Mean Squared Error
            dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (2 / n_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            cost = np.mean((y_pred - y) ** 2)
            self.cost_history.append(cost)

        return self

    def predict(self, X):
        """Makes predictions on new data vectors."""
        return np.dot(X, self.weights) + self.bias