"""Softmax model."""

import numpy as np


class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def sigmoid_calculate(self, x):
        if x < 0:
            return np.exp(x) / (1 + np.exp(x))
        else:
            return 1 / (1 + np.exp(-x))
    
    def softmax_calculate(self, x):
        exp = np.exp(x - np.max(x))
        return exp / np.sum(exp)
    
    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the softmax loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            gradient with respect to weights w; an array of same shape as w
        """
        N, D = X_train.shape
        grad_w = np.zeros_like(self.w)
        
        for i in range(N):
            x, y = X_train[i], y_train[i]
            if self.n_class == 2:
                # Binary classification with labels in {-1, 1}
                pred = np.dot(x, self.w)
                s = self.sigmoid_calculate(y * pred) 
                grad_w += - y * (1 - s) * x
            else:
                dot_prod = np.dot(x, self.w.T)
                prior_prob = self.softmax_calculate(dot_prod)
                for c in range(self.n_class):
                    if c == y:
                        grad_w[c] += (prior_prob[c] - 1) * x
                    else:
                        grad_w[c] += prior_prob[c] * x
        
        grad_w += self.reg_const * self.w
        return grad_w

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.
        - Initialize self.w as a matrix with random values sampled uniformly from [-1, 1)
        and scaled by 0.01. This scaling prevents overly large initial weights,
        which can adversely affect training.
        
        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        N, D = X_train.shape
        lr_decay = 1.2
        batch_size = 256

        X_train_orig = X_train.copy()
        y_train_orig = y_train.copy()

        # Augment training data 
        X_train = np.concatenate((np.ones((N, 1)), X_train), axis=1)
        
        np.random.seed(42)
        if self.n_class == 2:
            self.w = np.random.randn(D + 1)
        else:
            self.w = np.random.randn(self.n_class, D + 1)
            
        for epoch in range(self.epochs):
            indices = np.arange(N)
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]
            
            for batch in range(0, N, batch_size):
                X_batch = X_train[batch:batch + batch_size]
                y_batch = y_train[batch:batch + batch_size]
                grad_w = self.calc_gradient(X_batch, y_batch)
                self.w -= self.lr * grad_w
            
            preds = self.predict(X_train_orig)
            acc = np.mean(preds == y_train_orig)
            print("Epoch {}: training accuracy = {:.4f}".format(epoch + 1, acc))
            
            self.lr /= (1 + lr_decay * epoch)
            # self.lr *= 0.95

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        N, D = X_test.shape
        X_test = np.concatenate((np.ones((N, 1)), X_test), axis=1)
        
        if self.n_class == 2:

            # binary classification, so decision boundary at 0.
            y_pred = np.dot(X_test, self.w)
            y_pred = np.where(y_pred >= 0, 1, -1)
            return y_pred.astype(int)
        else:
            preds = np.zeros(N, dtype=int)
            for i in range(N):
                x = X_test[i]
                dot_prod = np.dot(x, self.w.T)
                prior_prob = self.softmax_calculate(dot_prod)
                preds[i] = np.argmax(prior_prob)
            return preds
