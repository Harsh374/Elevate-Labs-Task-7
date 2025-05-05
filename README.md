# Support Vector Machines (SVM) Implementation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview

This repository contains the implementation of Support Vector Machines (SVM) for Task 7 of the Elevate Labs Machine Learning Track. The project demonstrates both linear and non-linear SVM implementations on the Breast Cancer dataset, showcasing decision boundary visualization, hyperparameter tuning, and model evaluation.

## 🎯 Objective

The objective of this task is to:
- Implement SVMs for binary classification
- Compare linear and non-linear (RBF) kernel performance
- Visualize decision boundaries
- Tune hyperparameters for optimal performance
- Evaluate models using cross-validation

## 📊 Dataset

The implementation uses the [Breast Cancer Wisconsin (Diagnostic) Dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset), which contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The task is to classify tumors as malignant or benign.

### Dataset Features:
- 30 numeric, predictive attributes
- 1 target variable: diagnosis (M = malignant, B = benign)
- 569 instances

## 🛠️ Implementation Details

### Tools & Libraries Used
- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

### Key Components
1. **Data Preprocessing**
   - Handling missing values using SimpleImputer
   - Feature scaling with StandardScaler
   - Conversion of categorical target to binary

2. **SVM Implementation**
   - Linear SVM for linearly separable data
   - RBF kernel SVM for non-linear patterns

3. **Visualization**
   - PCA for dimensionality reduction
   - Decision boundary visualization
   - Learning curves

4. **Hyperparameter Tuning**
   - Grid search for C parameter
   - Gamma tuning for RBF kernel

5. **Model Evaluation**
   - Cross-validation assessment
   - Confusion matrix analysis
   - Feature importance visualization

## 🔍 Key Findings

- Comparison of performance between linear and non-linear kernels
- Optimal hyperparameter values for each kernel type
- Feature importance analysis
- Visualization of the decision boundaries
- Cross-validation performance metrics

## 📈 Results

The implementation achieves high accuracy in classifying breast cancer samples as malignant or benign. Detailed results include:
- Accuracy metrics for both kernels
- Precision, recall, and F1-scores
- Cross-validation stability assessment
- Identification of most important features

## 📚 Learning Outcomes

Through this implementation, the following concepts were explored:
- Margin maximization in SVMs
- The kernel trick for non-linear separation
- Effects of C parameter on regularization
- Handling overfitting in SVMs
- Feature scaling importance for SVMs
- Decision boundary visualization techniques

## 📝 File Structure

```
Elevate-Labs-Task-7/
├── breast-cancer.csv              # Dataset file
├── svm_implementation.ipynb       # Jupyter notebook with implementation
├── README.md                      # This file
└── results/                       # Generated visualizations
    ├── linear_decision_boundary.png
    ├── rbf_decision_boundary.png
    ├── feature_importance.png
    └── learning_curves.png
```

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/Harsh374/Elevate-Labs-Task-7.git
   cd Elevate-Labs-Task-7
   ```

2. Install requirements:
   ```bash
   pip install numpy pandas scikit-learn matplotlib
   ```

3. Open and run the Jupyter notebook:
   ```bash
   jupyter notebook svm_implementation.ipynb
   ```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- Elevate Labs for providing the task requirements
- The scikit-learn team for their excellent ML implementation
- UCI Machine Learning Repository for the original dataset

---

Created by [Harsh374](https://github.com/Harsh374) as part of the Elevate Labs Machine Learning Track.
