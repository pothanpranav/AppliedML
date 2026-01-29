Assignment 1: Prototype
SMS Spam Classification
Objective

The objective of this assignment is to build a prototype system that classifies SMS messages as Spam or Ham (Not Spam) using machine learning models. The system includes data preparation, model training, evaluation, and selection of the best-performing model.

1. Instructions Implementation
prepare.ipynb

The following functions were implemented:

✔ Load Data

A function was created to load SMS spam data from a tab-separated file and convert it into a Pandas DataFrame.

✔ Preprocess Data

Preprocessing steps applied:

Convert text to lowercase

Remove punctuation and special characters using regex

Remove extra whitespace

Convert labels:

Ham → 0

Spam → 1

✔ Split Data

Dataset split into:

Training Set → 70%

Validation Set → 15%

Test Set → 15%

Stratified sampling was used to maintain class distribution.

✔ Save Splits

Datasets saved as:

train.csv

validation.csv

test.csv

train.ipynb

The following functions and workflows were implemented:

✔ Model Training

Used TF-IDF vectorization with:

Multinomial Naive Bayes

Logistic Regression

Linear SVM

✔ Scoring Function

Evaluation metrics implemented:

Accuracy

Precision

Recall

F1 Score

✔ Model Validation

Each model was:

Trained on training data

Evaluated on training and validation datasets

Compared using performance metrics

✔ Model Selection

All three models were tested on the test dataset and best model selected using F1 score and overall performance.

2. Resources Used

Dataset:

UCI SMS Spam Collection Dataset

Learning Resources:

Radim Řehůřek Data Science Guide

Introduction to Statistical Learning (Chapters 1–3)

3. Solution Structure
prepare.ipynb Implementation
Dataset Loading

Data was loaded from file using Pandas with tab separator and encoding handling.

Data Preprocessing

Steps performed:

Lowercase conversion

Regex cleaning

Whitespace normalization

Label encoding

Data Splitting

Used stratified train-test split:

Train → 3900 samples

Validation → 836 samples

Test → 836 samples

Data Storage

Split datasets saved to CSV for reproducibility.

train.ipynb Implementation
Model Training Function

Models trained using Pipeline:

TF-IDF Vectorizer

Classifier

Model Evaluation Function

Used:

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

Classification Report

4. Model Performance
Training + Validation Performance Summary
Naive Bayes

Strength:

Very high precision (no false spam alerts)

Weakness:

Lower recall (misses some spam)

Logistic Regression

Strength:

High precision

Good accuracy

Weakness:

Lower recall than Naive Bayes

Linear SVM

Strength:

Highest accuracy

Highest recall

Best F1 score

Best generalization

5. Test Set Benchmark Results
Model	Accuracy	Precision	Recall	F1
Naive Bayes	97.37%	100%	80.36%	89.11%
Logistic Regression	96.29%	100%	72.32%	83.94%
Linear SVM	98.56%	99.02%	90.18%	94.39%
6. Final Model Selection

Best Model Selected: Linear SVM

Reason:

Highest F1 Score

Highest Accuracy

Strong Precision + Recall Balance

Best Test Performance

7. Final Test Performance (Linear SVM)

Accuracy: 99%

Ham Detection:

Precision: 0.99

Recall: 1.00

Spam Detection:

Precision: 0.99

Recall: 0.90

Confusion Matrix:

Ham correctly detected → 723

Spam correctly detected → 101

Spam missed → 11

False spam alerts → 1

8. Conclusion

The SMS Spam Classification prototype was successfully developed. Data was properly preprocessed, split, and stored. Three machine learning models were trained and evaluated. Among them, Linear SVM provided the best performance across all metrics and showed strong generalization ability.

Therefore, Linear SVM was selected as the final model for SMS spam classification.

