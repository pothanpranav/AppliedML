# Assignment 3: Testing & Model Serving

## Objective

The objective of this assignment is to:

-   Implement unit testing for a model scoring function.
-   Serve a trained machine learning model using Flask.
-   Write integration tests for the API endpoint.
-   Generate a test coverage report using pytest.

The trained model (`model.pkl`) exported from `train.ipynb` in
Assignment 2 is reused for scoring and testing.

------------------------------------------------------------------------

# Part 1: Unit Testing

## Score Function

In `score.py`, implement the following function:

``` python
def score(text: str,
          model,
          threshold: float = 0.5) -> (prediction: int, propensity: float):
```

### Function Responsibilities

The function should:

1.  Validate input types.
    -   `text` must be a string.
    -   `threshold` must be numeric and between 0 and 1.
2.  Compute probability using: `model.predict_proba([text])`
3.  Apply threshold-based classification.
4.  Return:
    -   `prediction` → 0 (HAM) or 1 (SPAM)
    -   `propensity` → probability score between 0 and 1

------------------------------------------------------------------------

## Unit Tests in `test.py`

The following test cases are implemented:

### 1. Smoke Test

-   Ensures the function executes without crashing.
-   Confirms correct return types.

### 2. Format Tests

-   Prediction must be integer.
-   Propensity must be float.
-   Propensity must lie between 0 and 1.

### 3. Prediction Sanity Check

-   Prediction must be either 0 or 1.

### 4. Threshold Edge Cases

-   If `threshold = 0`, prediction should always be 1.
-   If `threshold = 1`, prediction should always be 0.

### 5. Content-Based Tests

-   Obvious spam text → prediction = 1
-   Obvious non-spam text → prediction = 0

### 6. Exception Handling Tests

-   Non-string input → raises `TypeError`
-   Invalid threshold → raises `ValueError`

------------------------------------------------------------------------

# Part 2: Flask Model Serving

## Flask Application (`app.py`)

A Flask application is created to serve the trained model.

### Homepage `/`

-   Renders a minimal HTML form.
-   Accepts user input.
-   Displays SPAM / HAM label and propensity score.

### API Endpoint `/score`

-   Method: POST
-   Accepts:
    -   JSON → `{"text": "..."}`
    -   OR form-data → `text=...`

Returns:

``` json
{
    "prediction": 0 or 1,
    "propensity": float
}
```

Returns HTTP 400 if input text is missing or invalid.

------------------------------------------------------------------------

# Part 3: Integration Testing

Integration tests are implemented using Flask's `test_client()`.

### Integration Test Cases

-   Valid POST request to `/score`
-   JSON response validation
-   Status code check (200)
-   Error case when text is missing (400)

This ensures end-to-end API functionality and proper error handling.

------------------------------------------------------------------------

# Part 4: Coverage Report

Coverage is generated using:

``` bash
py -m pytest --cov=. --cov-report=term-missing > coverage.txt
```

## Coverage Output

    Name                  Stmts   Miss  Cover   Missing
    ---------------------------------------------------
    app.py                   18      1    94%   34
    score.py                 11      0   100%
    test_assignment3.py      48      0   100%
    ---------------------------------------------------
    TOTAL                    77      1    99%

## Remark

Line 34 in `app.py` corresponds to:

``` python
if __name__ == "__main__":
    app.run(...)
```

This block starts the Flask development server and is not executed
during pytest because the module is imported rather than run as a
standalone script. Therefore, it is intentionally not covered in the
test suite.

------------------------------------------------------------------------

# Files Included

-   `score.py`
-   `app.py`
-   `test.py`
-   `model.pkl`
-   `coverage.txt`

------------------------------------------------------------------------

# Final Outcome

✔ Complete unit testing\
✔ Integration testing for API\
✔ Exception handling covered\
✔ HTML interface + REST API\
✔ 99% coverage\
✔ Clean and modular design
