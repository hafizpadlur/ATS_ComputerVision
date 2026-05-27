# EMNIST Handwritten Character Recognition using HOG and SVM

## Overview

This project implements handwritten alphabet classification using the EMNIST Letters dataset with classical machine learning techniques. The system uses **Histogram of Oriented Gradients (HOG)** for feature extraction and **Support Vector Machine (SVM)** for classification.

The implementation includes:
- Dataset preprocessing
- Dataset balancing
- HOG feature extraction
- SVM classification
- Hyperparameter tuning using GridSearchCV
- Performance evaluation
- Result visualization

This project was developed as part of a **Computer Vision assignment**.

---

## Introduction

Handwritten character recognition is a fundamental problem in computer vision and pattern recognition. Variations in handwriting styles introduce challenges such as differences in shape, stroke thickness, and orientation.

This project applies:
- **Histogram of Oriented Gradients (HOG)** to capture edge and shape information
- **Support Vector Machine (SVM)** for multi-class classification

The combination of HOG and SVM provides an efficient and lightweight solution for handwritten character recognition tasks.

---

## Project Objectives

- Load and preprocess the EMNIST Letters dataset
- Balance dataset samples
- Extract image features using HOG
- Train an SVM classifier
- Perform hyperparameter tuning
- Evaluate model performance
- Generate confusion matrix visualization

---

## Dataset Information

### Dataset Used
- **EMNIST Letters Dataset**

Dataset source:
- Kaggle EMNIST Dataset

Dataset is automatically downloaded using:
```python
kagglehub.dataset_download("crawford/emnist")
```

---

### Dataset Characteristics

| Property          | Value     |
|------------------|-----------|
| Number of Classes | 26        |
| Labels            | A–Z       |
| Image Size        | 28 × 28   |
| Image Type        | Grayscale |
| Data Format       | CSV       |

---

## Sampling Strategy

Balanced sampling is applied to reduce computational cost while maintaining equal class distribution.

Configuration:
- 30 samples per class
- Total classes: 26

Total samples:
```
26 × 30 = 780 samples
```

---

## Libraries Used

### Data Processing
- NumPy
- Pandas

### Visualization
- Matplotlib
- Seaborn

### Feature Extraction
- scikit-image

### Machine Learning
- scikit-learn

### Dataset Download
- kagglehub

---

## Installation

```bash
pip install numpy pandas matplotlib seaborn kagglehub scikit-image scikit-learn
```

---

## Project Structure

```
ATS_ComputerVision/
│
├── main.py
├── README.md
│
└── results/
    ├── sample_dataset.png
    ├── confusion_matrix.png
    └── classification_report.txt
```

---

## Conclusion

This project demonstrates that combining **Histogram of Oriented Gradients (HOG)** and **Support Vector Machine (SVM)** can effectively classify handwritten alphabet characters from the EMNIST dataset with low computational cost.
