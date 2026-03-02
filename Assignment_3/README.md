# Assignment 3 --- Testing & Model Serving

This repository implements a spam text classifier scoring pipeline along
with unit tests, Flask model serving, and coverage reporting.

The project demonstrates how a trained machine learning model (exported
as `model.pkl`) can be packaged into a reusable scoring function, tested
using `pytest`, and exposed through a Flask API endpoint with an
optional HTML interface.

------------------------------------------------------------------------

## Features

-   Reusable `score()` function for model inference
-   Input validation and threshold-based prediction
-   Comprehensive unit tests covering:
    -   Smoke testing
    -   Type validation
    -   Edge cases
    -   Content-specific cases
    -   Exception handling
-   Flask application with:
    -   HTML interface (`/`)
    -   REST API endpoint (`/score`)
-   Integration testing using Flask test client
-   Automated coverage reporting using `pytest-cov`
-   99% total test coverage

------------------------------------------------------------------------

## Project Structure

    Assignment3/
    │
    ├── app.py              # Flask application
    ├── score.py            # Model scoring logic
    ├── test.py             # Unit + integration tests
    ├── model.pkl           # Trained ML model (from Assignment 2)
    ├── coverage.txt        # Coverage report
    ├── Question.md         # Assignment description
    └── README.md           # Project documentation

------------------------------------------------------------------------

## How to Run

### 1. Install Dependencies

``` bash
pip install flask pytest pytest-cov scikit-learn
```

------------------------------------------------------------------------

### 2. Run Unit & Integration Tests

``` bash
py -m pytest
```

------------------------------------------------------------------------

### 3. Generate Coverage Report

``` bash
py -m pytest --cov=. --cov-report=term-missing > coverage.txt
```

------------------------------------------------------------------------

### 4. Run Flask Application

``` bash
python app.py
```

Open in browser:

    http://127.0.0.1:5000

------------------------------------------------------------------------

## API Endpoint

### POST `/score`

Accepts:

-   JSON:

```{=html}
<!-- -->
```
    {"text": "Free lottery ticket now!"}

OR

-   Form-data:

```{=html}
<!-- -->
```
    text=Free lottery ticket now!

Returns:

    {
      "prediction": 0 or 1,
      "propensity": float
    }

Returns HTTP 400 if input text is missing.

------------------------------------------------------------------------

## Coverage Summary

    Name                  Stmts   Miss  Cover
    -----------------------------------------
    app.py                   18      1    94%
    score.py                 11      0   100%
    test.py                  XX      0   100%
    -----------------------------------------
    TOTAL                    99%

Remark:

Line corresponding to the `if __name__ == "__main__"` block in `app.py`
is not covered because pytest imports the module instead of executing it
as a standalone script. This is expected behavior.

------------------------------------------------------------------------

## References

-   Pytest documentation: https://docs.pytest.org
-   Flask documentation: https://flask.palletsprojects.com
-   Scikit-learn documentation: https://scikit-learn.org

------------------------------------------------------------------------

## Final Outcome

✔ Clean separation of scoring logic and API layer\
✔ Robust unit and integration testing\
✔ Error handling and edge-case validation\
✔ RESTful API + HTML interface\
✔ 99% test coverage\
✔ Production-ready modular design
