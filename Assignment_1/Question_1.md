# Assignment 1: Prototype (Due 30 Jan 2026)

## Build a Prototype for SMS Spam Classification

Develop a prototype for classifying SMS messages as spam or not spam.
The system includes data preparation, model training, evaluation, and
final model selection.

------------------------------------------------------------------------

## Instructions

### `prepare.ipynb`

Write functions to:

-   Load the data from a given file path
-   Preprocess the data (if needed)
-   Split the data into train/validation/test sets
-   Store the splits as `train.csv`, `validation.csv`, and `test.csv`

------------------------------------------------------------------------

### Implementation

#### Load Data

The dataset is loaded from a tab-separated file using Pandas. Missing
values are removed and the dataset is converted into a structured
DataFrame.

#### Preprocess Data

Preprocessing steps applied: - Convert text to lowercase
- Remove punctuation and special characters using regex
- Remove extra whitespace
- Convert labels: - Ham → 0
- Spam → 1

#### Split Data

The dataset is split into: - Training Set → 70%
- Validation Set → 15%\
- Test Set → 15%

Stratified sampling is used to maintain class balance.

#### Store Splits

The split datasets are saved as: - `train.csv`
- `validation.csv`
- `test.csv`

------------------------------------------------------------------------

## `train.ipynb`

Write functions to:

-   Fit a model on the training data
-   Score a model on a given dataset
-   Evaluate the model predictions

------------------------------------------------------------------------

### Validate the Model

-   Fit the model on training data
-   Score on training and validation datasets
-   Evaluate performance using:
    -   Accuracy
    -   Precision
    -   Recall
    -   F1 Score
-   Fine-tune hyperparameters (if necessary)

------------------------------------------------------------------------

## Models Used

Three benchmark models were implemented:

-   Multinomial Naive Bayes
-   Logistic Regression
-   Linear Support Vector Machine (SVM)

TF-IDF Vectorization was used for feature extraction.

------------------------------------------------------------------------

## Model Validation Results

### Naive Bayes

Strength: - Very high precision (no false spam alerts)

Weakness: - Lower recall (misses some spam messages)

------------------------------------------------------------------------

### Logistic Regression

Strength: - High precision\
- Good overall accuracy

Weakness: - Lower recall compared to Naive Bayes

------------------------------------------------------------------------

### Linear SVM

Strength: - Highest accuracy
- Highest recall
- Best F1 score
- Best generalization

------------------------------------------------------------------------

## Test Set Benchmark Results

  Model                 Accuracy     Precision   Recall       F1
  --------------------- ------------ ----------- ------------ ------------
  Naive Bayes           97.37%       100%        80.36%       89.11%
  Logistic Regression   96.29%       100%        72.32%       83.94%
  Linear SVM            **98.56%**   99.02%      **90.18%**   **94.39%**

------------------------------------------------------------------------

## Final Model Selection

**Selected Model: Linear SVM**

Reason: - Highest F1 Score
- Highest Accuracy
- Strong Precision + Recall Balance
- Best Test Performance

------------------------------------------------------------------------

## Final Test Performance (Linear SVM)

Accuracy: **99%**

Ham Detection: - Precision: 0.99
- Recall: 1.00

Spam Detection: - Precision: 0.99
- Recall: 0.90

Confusion Matrix Summary: - Ham correctly classified → 723
- Spam correctly classified → 101
- Spam missed → 11
- False spam alerts → 1

------------------------------------------------------------------------


## Data Loading

The preprocessed datasets generated in `prepare.ipynb` are loaded:

- `train.csv`
- `validation.csv`
- `test.csv`

Each dataset contains:
- `message` → SMS text
- `label` → 0 (ham), 1 (spam)

Missing values are removed to ensure stable model training.

------------------------------------------------------------------------

## Feature Extraction

Text messages are converted into numerical features using:

- **TF-IDF Vectorization**
  - Removes English stopwords
  - Produces sparse, high-dimensional representations
  - Suitable for linear text classification models

------------------------------------------------------------------------

## Models Used

Three classification models are trained and evaluated:

- **Multinomial Naive Bayes**
- **Logistic Regression**
- **Linear Support Vector Machine (SVM)**

All models are implemented using a unified **Pipeline** structure that
combines TF-IDF vectorization with classification.

------------------------------------------------------------------------

## Baseline Model Training

Each model is first trained using default hyperparameters.

For each baseline model:
- Training and validation predictions are generated
- Evaluation metrics are computed:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Confusion matrices are plotted
- ROC curves are plotted on the validation set

Baseline models provide a reference point before tuning.

------------------------------------------------------------------------

## Hyperparameter Tuning

Hyperparameter tuning is performed using **GridSearchCV** with
5-fold cross-validation.

Tuned parameters:

- **Naive Bayes**
  - `alpha`
- **Logistic Regression**
  - `C`
- **Linear SVM**
  - `C`

The **F1-score** is used as the optimization metric due to class imbalance.

------------------------------------------------------------------------

## Tuned Model Evaluation

After tuning, all models are evaluated again on:

- Training dataset
- Validation dataset

For each tuned model:
- Accuracy, precision, recall, and F1-score are reported
- Confusion matrices are plotted
- ROC curves are plotted on validation data

------------------------------------------------------------------------

## Threshold Analysis

For the Linear SVM model:

- Decision scores are extracted using `decision_function`
- Precision–Recall curve is computed
- Spam-class F1-score is analyzed across decision thresholds

This analysis helps understand the trade-off between precision and recall.

------------------------------------------------------------------------

## Feature Interpretability

Feature weights from the trained Linear SVM model are analyzed.

- Words with highest positive weights → strong spam indicators
- Words with most negative weights → strong ham indicators

This provides insight into which terms influence predictions.

------------------------------------------------------------------------

## Learning Curve Analysis

A learning curve is plotted for the Linear SVM model:

- Training sizes range from 20% to 100%
- 5-fold cross-validation is used
- F1-score is reported for both training and validation sets

This helps assess:
- Model generalization
- Bias–variance behavior
- Effect of increasing training data size

------------------------------------------------------------------------

## Model Selection

Models are compared using **validation F1-score**.

The model with the highest validation F1-score is selected as the final model.
This ensures that model selection is based only on unseen validation data.

------------------------------------------------------------------------

## Test Set Evaluation

The selected model is evaluated once on the test dataset.

Evaluation includes:
- Classification report
- Confusion matrix
- ROC curve

The test set is not used during training or tuning.

------------------------------------------------------------------------

## Final Outcome

**Selected Model:** Linear Support Vector Machine (SVM)

The Linear SVM achieves:
- High accuracy
- Strong precision and recall balance
- Best F1-score on validation and test set




## Conclusion

The SMS Spam Classification prototype was successfully developed. Data
preprocessing, dataset splitting, model training, and evaluation were
completed. Among all tested models, Linear SVM achieved the best overall
performance and generalization ability. Therefore, Linear SVM is
selected as the final model for deployment.
