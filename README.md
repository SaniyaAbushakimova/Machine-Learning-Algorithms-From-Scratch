# About

This repository contains implementations of various **Machine Learning and Statistical Learning algorithms from scratch**, developed as part of the **Practical Statistical Learning** course. Each project focuses on building models without relying on high-level machine learning libraries, providing deeper insights into their mathematical foundations and optimizations.

Each folder contains:
* ipynb files with implementation (Python);
* Corresponding datasets;
* Instructions on implementation details.

## Implemented Algorithms and Projects

### GMM-and-HMM-with-Expectation-Maximization
Gaussian Mixture Models (GMM) and Hidden Markov Models (HMM) using the EM Algorithm. \
Project completed on October 20, 2024.

* Implemented Expectation-Maximization (EM) from scratch to fit GMMs for clustering and density estimation.
* Developed Baum-Welch (EM for HMMs) and Viterbi Algorithm to train and decode Hidden Markov Models.
* Applied the models to sequence modeling and probabilistic clustering.

### KNN-and-Bayes-Classification
Comparing k-Nearest Neighbors (kNN) and Bayes Rule for Classification. \
Project completed on September 6, 2024.

* Implemented custom kNN classifier with cross-validation for hyperparameter selection.
* Developed Bayes Classifier from scratch, leveraging probability distributions for decision-making.
* Conducted a simulation study to compare kNN and Bayes decision rules in different distributions.

### LOESS-RidgelessRegression-NCS
Nonparametric Regression and Overfitting in High-Dimensional Models. \
Project completed on September 30, 2024.

* Implemented LOESS (Locally Weighted Scatterplot Smoothing) for nonlinear regression.
* Explored Ridgeless Regression to analyze overfitting and the Double Descent phenomenon.
* Used Natural Cubic Splines (NCS) for time series smoothing and feature extraction.

### Lasso-with-Coordinate-Descent
Sparse Regression with L1 Regularization. \
Project completed on September 18, 2024.

* Implemented Lasso Regression from scratch using the Coordinate Descent algorithm.
* Compared Lasso with Ridge Regression and Principal Component Regression (PCR).
* Analyzed model sparsity and feature selection using simulated datasets.

### SVM-with-Pegasos-Algorithm
Support Vector Machines (SVM) using a Specialized SGD Method. \
Project completed on November 12, 2024.

* Developed Support Vector Machines (SVM) from scratch, solving the primal form directly.
* Implemented Pegasos Algorithm (Primal Estimated sub-GrAdient SOlver for SVM), a variation of Stochastic Gradient Descent (SGD) optimized for large-scale datasets.
* Applied the model to binary classification tasks on MNIST subsets and evaluated generalization performance.
