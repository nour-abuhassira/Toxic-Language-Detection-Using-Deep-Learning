# Toxic Language Detection Using Deep Learning

## Project Overview
This project focuses on identifying and classifying types of toxicity in text data using machine learning techniques. In addition to building a baseline classification pipeline, this repository includes an analytical comparison of first-order versus second-order optimization methods to understand learning convergence behavior.

---

## Key Data Insights & Challenges

### 1. Class Imbalance
During the Exploratory Data Analysis (EDA) phase, target label distributions were thoroughly evaluated. The analysis revealed a severe class imbalance, with **"toxic"** identified as the most frequent class among all labeled categories. Standard accuracy metrics can be misleading under these conditions; therefore, model performance should be evaluated using robust metrics like F1-score or precision-recall curves.

### 2. Feature Overlap Limitations
The baseline text classification model utilizes a **Linear SVC** classifier paired with **n-gram features**. While computationally efficient, evaluation highlighted a distinct performance bottleneck caused by **feature overlap discrepancies**. Because different text categories share a heavily overlapping vocabulary, the linear boundary struggles to isolate intent and nuance. This underscores the necessity of moving toward semantic, context-aware embeddings (such as transformers) for complex classification tasks.

---

## Optimization Analysis: First-Order vs. Second-Order

A component of this study involves comparing how different mathematical optimization frameworks update model parameters during training[cite: 1]:

* **First-Order Optimization (Backpropagation):** Relies strictly on the gradient vector (first derivatives) to update weights[cite: 1]. It features low computational overhead per iteration and scales seamlessly to massive architectures, though it requires careful learning rate tuning[cite: 1].
* **Second-Order Optimization (Newton's Method):** Incorporates the Hessian matrix (second derivatives) to account for the curvature of the loss surface[cite: 1]. While it offers quadratic convergence and more precise step trajectories, the $O(N^3)$ computational complexity of calculating and inverting the Hessian makes it restrictive for high-dimensional feature spaces[cite: 1].

---

## Model Pipeline & Architecture

1. **Exploratory Data Analysis:** Feature distribution mapping and class frequency balancing.
2. **Text Preprocessing:** Tokenization, n-gram extraction, and TF-IDF vectorization.
3. **Classification Baseline:** Linear Support Vector Classification (SVC).
4. **Optimization Evaluation:** Convergence tracking and metric comparisons (including learning behavior and error metrics)[cite: 1].

---

## Getting Started

### Prerequisites
* Python 3.x
* scikit-learn
* pandas / numpy

### Running the Pipeline
To execute the baseline model training and view the classification report:
```bash
python src/train.py
