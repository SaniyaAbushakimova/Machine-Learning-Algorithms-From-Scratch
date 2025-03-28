"""Logistic regression model."""

import numpy as np


class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        positive = z >= 0
        negative = ~positive
        result = np.empty_like(z)
        
        result[positive] = 1 / (1 + np.exp(-z[positive]))
        # For negative z, to avoid overflow in exp(-z), we do following: 
        exp_z = np.exp(z[negative])
        result[negative] = exp_z / (1 + exp_z)

        return result

    

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        - Use the logistic regression update rule as introduced in lecture.
        - Initialize self.w as a matrix with random values sampled uniformly from [-1, 1)
        and scaled by 0.01. 
        - This initialization prevents the weights from starting too large,
        which can cause saturation of the sigmoid function 

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        N, D = X_train.shape
        X_train = np.concatenate((np.ones((N, 1)), X_train), axis=1)
        # Initialize random weights
        self.w = np.random.randn(D + 1)
        # labels from {-1, 1} -> {0, 1} 
        y_train_converted = (y_train + 1) / 2

        for epoch in range(self.epochs):
            for i in range(N):
                pred = np.dot(X_train[i], self.w)
                sigmoid_val = self.sigmoid(pred)
                grad_w = (sigmoid_val - y_train_converted[i]) * X_train[i]
                self.w -= self.lr * grad_w
        return self.w

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:exce
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        N, D = X_test.shape
        X_test = np.concatenate((np.ones((N, 1)), X_test), axis=1)
        y_pred = np.dot(X_test, self.w)
        y_pred = self.sigmoid(y_pred)
        # Convert probabilities to {-1, 1}
        return np.where(y_pred > self.threshold, 1, -1)

