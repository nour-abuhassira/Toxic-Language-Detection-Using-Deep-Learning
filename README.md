# Toxic Language Detection Using Deep Learning

This project focuses on detecting and classifying toxic online comments using both traditional machine learning and state-of-the-art deep learning models. By analyzing textual content from online discussions, the project develops robust classification systems capable of identifying harmful language while providing explainable and trustworthy predictions.

## Project Overview

Online platforms face significant challenges in moderating toxic content, including insults, threats, hate speech, and abusive language. This project leverages a toxic comments dataset to build and evaluate multiple machine learning and deep learning approaches for automated toxicity detection.

The project compares traditional Natural Language Processing (NLP) techniques with modern transformer-based architectures to determine the most effective solution for toxic language classification.

The evaluated models include:

* TF-IDF + Logistic Regression (Baseline)
* LSTM (Long Short-Term Memory)
* Bidirectional LSTM
* DistilBERT Transformer Model

In addition to model performance, the project emphasizes:

* Model explainability using LIME
* Error and failure analysis
* Robustness evaluation
* Hyperparameter optimization
* Comparative performance assessment

## Project Structure

### Data Preprocessing & Exploratory Analysis

The initial phase focuses on understanding and preparing the textual data:

#### Exploratory Data Analysis (EDA)

* Analyzing class distributions and toxicity prevalence.
* Investigating comment length patterns and linguistic characteristics.
* Identifying class imbalance issues.

#### Text Preprocessing

* Text cleaning and normalization.
* Tokenization and sequence preparation.
* TF-IDF vectorization for traditional machine learning models.
* Text encoding for deep learning architectures.

#### Data Preparation

* Train-validation-test splitting.
* Handling class imbalance.
* Feature engineering and preprocessing pipelines.

### Machine Learning & Deep Learning Modeling

The core of the project involves building and comparing multiple classification approaches:

#### Baseline Model

* TF-IDF + Logistic Regression

#### Deep Learning Models

* LSTM Network
* Bidirectional LSTM Network

#### Transformer-Based Model

* DistilBERT Fine-Tuning for Toxic Language Classification

#### Hyperparameter Optimization

* Model tuning and performance optimization.
* Evaluation across multiple metrics.

### Model Evaluation

Performance is assessed using several classification metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

The project includes a detailed comparison between traditional machine learning methods and deep learning architectures to identify the most effective model.

### Explainability & Interpretability

To improve transparency and trustworthiness:

#### LIME Explanations

* Local Interpretable Model-Agnostic Explanations (LIME) are used to understand individual predictions.
* Identification of words and phrases that contribute most strongly to toxicity classifications.

#### Feature Importance Analysis

* Understanding which textual patterns influence model decisions.

### Robustness & Failure Analysis

The project evaluates model reliability through:

#### Failure Analysis

* Examination of incorrectly classified comments.
* Identification of common error patterns.

#### Robustness Testing

* Evaluation of model behavior under challenging inputs.
* Assessment of generalization capabilities.

## Dataset Variables

The dataset consists of online user comments and associated toxicity labels.

### Input Features

* Raw textual comments.
* Linguistic and semantic information extracted through NLP techniques.

### Target Variable

**Toxicity Label**

* 0 = Non-Toxic Comment
* 1 = Toxic Comment

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* TensorFlow / Keras
* Hugging Face Transformers
* DistilBERT
* LIME

## How to Run

To replicate the analysis and model training, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/nour-abuhassira/Toxic-Language-Detection.git
cd Toxic-Language-Detection
```

### 2. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow transformers lime
```

### 3. Execute the Notebook

Open and run:

```bash
Toxic_Comments.ipynb
```

to view the exploratory analysis, model development process, evaluation results, explainability analysis, and robustness assessment.

## Key Outcomes

* Built and compared traditional machine learning and deep learning approaches for toxicity detection.
* Fine-tuned DistilBERT for advanced text classification.
* Applied Explainable AI (XAI) techniques using LIME.
* Conducted failure analysis and robustness testing.
* Evaluated model performance across multiple architectures and metrics.
* Identified the most effective architecture for toxic language classification while maintaining model interpretability.
