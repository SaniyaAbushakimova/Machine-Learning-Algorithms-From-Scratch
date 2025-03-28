"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
    def __init__(self, n_features: int, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        np.random.seed(42)
        
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_features = n_features # = features + bias
        self.n_class = n_class
        
        # 1. Binary classification: one column
        if self.n_class == 2:
         self.w = np.random.uniform(-1, 1, size=(self.n_features, 1)) * 0.01
          
        # 2. Multi-class classification: one row per class
        else:
         self.w = np.random.uniform(-1, 1, size=(self.n_features, self.n_class)) * 0.01

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        logits = np.dot(X_train, self.w)
        gradient = np.zeros(self.w.shape, dtype='float64') 
        
        # 1. Binary classification:
        if self.n_class == 2:
            margin = y_train * logits.flatten()
            incorrects_mask = margin < 1
            
            # Update the incorrect class
            if np.any(incorrects_mask):
                X_incorrects = X_train[incorrects_mask]
                y_incorrects = y_train[incorrects_mask].reshape(-1, 1)
                
                # Compute gradient 
                gradient = (-y_incorrects * X_incorrects).sum(axis=0).reshape(-1, 1)
            
        # 2. Multi-class classification
        else:
            # 1) Find correct class scores/logits
            correct_class_scores = np.zeros((len(y_train), 1))
            for i in range(len(y_train)):
                correct_class_scores[i] = logits[i, y_train[i]] # y_train[i] - gives a correct class index
            
            # 2) Compute margin violations (Hinge Loss term)
            margins = logits - correct_class_scores + 1
            for i in range(len(y_train)):
                margins[i, y_train[i]] = 0  # Avoid correct class score from being penalized
            
            # 3) Find classes that violate margin constraint
            incorrects_mask = margins > 0
            incorrects_counts = incorrects_mask.sum(axis=1, keepdims=True)

            # 4) Iterate over each training sample
            for i in range(len(y_train)):
                for c in range(self.n_class):
                    if incorrects_mask[i, c]:
                        # 4.1) Update incorrect class
                        gradient[:, c] += X_train[i]  
            
                # 4.2) Update the correct class
                gradient[:, y_train[i]] -= incorrects_counts[i] * X_train[i]
            
        # Add L2 regularization term
        gradient = gradient/len(y_train) + (self.reg_const / len(y_train)) * self.w
        
        return gradient
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, batch_size: int = 32):
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
        
        for epoch in range(self.epochs):
            # Shuffle data
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train, y_train = X_train[indices], y_train[indices]
            
            # Divide data into batches and process eahc batch separately
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
            
                gradient = self.calc_gradient(X_batch, y_batch)
                self.w -= self.lr * gradient
            
            accuracy = self.get_acc(self.predict(X_train), y_train)
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
            return np.argmax(logits, axis=1) # (N, )     
            
    
    def get_acc(self, pred, y_test):
        return np.sum(y_test == pred) / len(y_test) * 100
