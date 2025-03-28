"""Perceptron model."""

import numpy as np

class Perceptron:
    def __init__(self, n_features: int, n_class: int, lr: float, epochs: int):
      """Initialize a new classifier.

      Parameters:
        n_features: the number of features
        n_class: the number of classes
        lr: the learning rate
        epochs: the number of epochs to train for
      """
      np.random.seed(42)
      
      self.n_features = n_features # = features + bias
      self.n_class = n_class
      self.lr = lr
      self.epochs = epochs

      # 1. Binary classification: one column
      if self.n_class == 2:
       self.w = np.random.uniform(-1, 1, size=(self.n_features, 1)) * 0.01
      
      # 2. Multi-class classification: one column per class
      else:
       self.w = np.random.uniform(-1, 1, size=(self.n_features, self.n_class)) * 0.01

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
      """Train the classifier.

      - Use the perceptron update rule as introduced in the Lecture.
      - Initialize self.w as a matrix with random values sampled uniformly from [-1, 1)
      and scaled by 0.01. This scaling prevents overly large initial weights,
      which can adversely affect training.

      Parameters:
        X_train: a number array of shape (N, D) containing training data;
        N examples with D dimensions
        y_train: a numpy array of shape (N,) containing training labels
      """
     
      for epoch in range(self.epochs):
        
        y_pred = self.predict(X_train)
        
        # 1. Binary classification: update weight vector when y_train is classified incorrectly
        if self.n_class == 2:
          # Identify instances that were predicted incorrectly
          incorrects_mask = (y_train != y_pred)

          if np.any(incorrects_mask):
            X_incorrects = X_train[incorrects_mask] # (N, D+1)
            y_incorrects = y_train[incorrects_mask].reshape(-1, 1) # (N, 1)

            # Updating weigths accordingly 
            self.w += self.lr * np.dot(X_incorrects.T, y_incorrects)

        # 2. Multi-class classification: update weight vectors of both y_train class and y_pred class when y_train is classified incorrectly
        else:
          # Identify misclassified instances and get class indices (y_train, y_pred)
          for i in range(X_train.shape[0]):
            # y_train \in [0, self.n_class - 1]
            # y_pred \in [0, self.n_class - 1]
            if y_train[i] != y_pred[i]:
              self.w[:, y_train[i]] += self.lr * X_train[i]
              self.w[:, y_pred[i]] -= self.lr * X_train[i]
        
        accuracy = self.get_acc(y_pred, y_train)
        print(f"Epoch {epoch + 1}/{self.epochs}, Training accuracy: {accuracy:.2f}%, Lr: {self.lr:.4f}")
        
        # Learning rate decay
        self.lr *= 0.95

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
      
      # Predicting labels
      logits = np.dot(X_test, self.w)

      # 1. Binary classification: return predicted class {-1, +1}
      if self.n_class == 2:
        return np.sign(logits).flatten() # (N, )

      # 2. Multi-class classification: return predicted class index [0, self.n_class - 1]
      else:
        return np.argmax(logits, axis = 1) # (N, )     
        
    def get_acc(self, pred, y_test):
        return np.sum(y_test == pred) / len(y_test) * 100