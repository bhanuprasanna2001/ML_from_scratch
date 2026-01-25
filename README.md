# Machine Learning from Scratch

This repository contains implementations of classical ML algorithms with detailed mathematical foundations and visualizations.

## Overview

This project implements core machine learning algorithms without relying on high-level ML libraries (scikit-learn, TensorFlow, etc.). Each implementation includes mathematical derivations, algorithmic explanations, and practical demonstrations.

For in blog format refer to: [BLOG ML](https://bhanuprasanna2001.github.io/learning/ai/ML/)

## Implemented Algorithms

### Supervised Learning

**Regression**
- Linear Regression (Closed-Form and Gradient Descent)
- Support Vector Regression (Subgradient Descent)
- K-Nearest Neighbors Regression (Weighted and Unweighted)

**Classification**
- Logistic Regression (Gradient Descent)
- Gaussian Naive Bayes
- K-Nearest Neighbors (Weighted and Unweighted)
- Support Vector Machines (SMO, CVXOPT, Subgradient Descent)
- Multi-class SVM (One-vs-Rest)
- Decision Tree Classifier (CART with Gini/Entropy)
- Random Forest Classifier

**Ensemble Methods**
- Random Forest (Classification and Regression)
- Gradient Boosting (Classification and Regression)

### Unsupervised Learning

**Clustering**
- K-Means

### Dimensionality Reduction & Linear Algebra
- Principal Component Analysis (PCA)
- Singular Value Decomposition (SVD) with Image Compression
- Jacobi Eigenvalue Algorithm
- Custom Matrix Multiplication (Simple) -> ikj is better than ijk

## Key Features

- **Pure NumPy implementations** with no ML framework dependencies
- **Mathematical rigor** with detailed comments explaining theory
- **Visualization tools** for model performance and decision boundaries
- **Multiple optimization approaches** (closed-form, gradient descent, SMO)
- **Custom linear algebra utilities** including Jacobi eigenvalue decomposition

## Project Structure

```
├── Linear Regression/       # Closed-form and gradient descent implementations
├── Logistic Regression/     # Binary classification with gradient descent
├── KNN/                     # K-nearest neighbors for classification and regression
├── K-Means/                 # Clustering algorithm
├── Naive Bayes/             # Gaussian Naive Bayes classifier
├── SVM/                     # Multiple SVM implementations (SMO, CVXOPT, subgradient)
├── Decision Tree/           # CART algorithm for classification and regression
├── Random Forest/           # Ensemble method with bagging
├── Gradient Boosting/       # Sequential ensemble learning
├── Bayesian/                # Bayesian methods exploration
└── Utils/                   # Linear algebra utilities (PCA, SVD, Jacobi, matrix ops)
```

## References

- [ML From Scratch by Erik Linder-Norén](https://github.com/eriklindernoren/ML-From-Scratch)
- Stanford CS229 Materials (Sequential Minimal Optimization)
- Educational YouTube channels: 3Blue1Brown, StatQuest, Ritvikmath, Normalized Nerd
- Various Medium articles and academic papers

## License

Open.
