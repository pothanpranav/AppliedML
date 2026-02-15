# Assignment 2: Experiment Tracking (Due 15 Feb 2026)

## Data Version Control

In `prepare.ipynb`, data versioning is implemented using **DVC (Data
Version Control)**.

### Steps Performed

-   Raw dataset loaded and stored as `raw_data.csv`
-   Dataset split into:
    -   `train.csv`
    -   `validation.csv`
    -   `test.csv`
-   Splits created using stratified sampling
-   Initial split created with `random_state=42`
-   Updated split created with `random_state=21`

### Version Tracking

-   `raw_data.csv` tracked using `dvc add`
-   Split files tracked using DVC
-   Changes committed using Git
-   Data pushed to DVC remote storage

### Checkout and Distribution Verification

Two versions of the dataset splits were verified:

1.  **Version 1 (seed=42)**
    -   Checked out using:
        -   `git checkout HEAD~1`
        -   `dvc checkout`
    -   Printed distribution of target variable (0s and 1s)
    -   Verified that train/validation/test splits follow class balance
2.  **Version 2 (seed=21)**
    -   Checked out using:
        -   `git checkout master`
        -   `dvc checkout`
    -   Printed distribution of target variable
    -   Confirmed that splits differ from Version 1

Verification Result: - Train sets are not identical - First SMS entries
differ - Data versions successfully tracked

### Bonus: Decoupling Compute and Storage

-   Google Drive used as DVC remote storage
-   Remote configured using:
    -   `dvc remote add -d storage <Drive path>`
-   Enables separation of computation and data storage

------------------------------------------------------------------------

## Model Version Control and Experiment Tracking

In `train.ipynb`, **MLflow** is used for experiment tracking and model
version control.

### MLflow Setup

-   Tracking URI set to Google Drive
-   Experiment created: `SMS_Spam_Classification`
-   All runs logged under this experiment

------------------------------------------------------------------------

## Benchmark Models Built and Tracked

Three benchmark models were implemented and tracked:

1.  **Naive_Bayes_SMS**
    -   CountVectorizer
    -   MultinomialNB (alpha=0.1)
2.  **Logistic_Regression_SMS**
    -   CountVectorizer
    -   LogisticRegression (max_iter=1000)
3.  **Random_Forest_SMS**
    -   CountVectorizer
    -   RandomForestClassifier (max_depth=60)

------------------------------------------------------------------------

## Metrics Logged

For each model, the following metrics were logged:

-   Validation Accuracy
-   Validation Precision
-   Validation Recall
-   Validation F1-score
-   Validation AUCPR
-   Test Accuracy
-   Test Precision
-   Test Recall
-   Test F1-score
-   Test AUCPR

Confusion matrix logged as artifact.

Models registered in MLflow Model Registry.

------------------------------------------------------------------------

## Model Selection Metric: AUCPR

Validation AUCPR values:

-   Naive_Bayes_SMS → 0.9631
-   Logistic_Regression_SMS → 0.9653
-   Random_Forest_SMS → 0.9772

Sorted by Validation AUCPR:

1.  Random_Forest_SMS → 0.9772
2.  Logistic_Regression_SMS → 0.9653
3.  Naive_Bayes_SMS → 0.9631

------------------------------------------------------------------------

## Final Model Selected

**Selected Model: Random_Forest_SMS**

Reason: - Highest Validation AUCPR - Strong generalization performance -
Best ranking of spam messages

------------------------------------------------------------------------

## Conclusion

This assignment demonstrates:

-   Proper data version control using DVC
-   Multiple dataset versions tracked and compared
-   Remote storage configuration using Google Drive
-   Full experiment tracking using MLflow
-   Model registration and metric comparison
-   Clean validation-based model selection using AUCPR

Both data and models are version-controlled, reproducible, and properly
tracked.
