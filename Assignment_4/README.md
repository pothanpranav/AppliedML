# Assignment 4 --- Containerization & Continuous Integration

**Course Assignment**\
**Date:** 17 Feb 2026\
**Due:** 8 Mar 2026

This assignment extends Assignment 3 by adding **Docker
containerization** and **Continuous Integration using Git hooks** for
the Flask spam classification application.

The goal is to demonstrate how a trained machine learning model can be
deployed inside a container and automatically tested during development.

------------------------------------------------------------------------

# Project Overview

The project contains a spam message classification system with the
following components:

-   A **model scoring function** (`score.py`)
-   A **Flask web application** (`app.py`) for serving predictions
-   **Unit and integration tests** using `pytest`
-   **Docker containerization**
-   **Docker integration testing**
-   **Continuous Integration using a Git pre‑commit hook**
-   **Coverage report generation**

------------------------------------------------------------------------

# Project Structure

    AML/
    │
    ├── app.py                 Flask application serving predictions
    ├── score.py               Model scoring function
    ├── model.pkl              Trained spam classification model
    ├── test_assignment4.py    Unit tests, API tests, and Docker test
    │
    ├── Dockerfile             Instructions for building Docker container
    ├── requirements.txt       Python dependencies
    │
    ├── coverage.txt           Pytest coverage report
    │
    └── hooks/
        └── pre-commit         Git pre‑commit hook for CI

------------------------------------------------------------------------

# Containerization

A **Docker container** is created for the Flask application.

The Docker container includes:

-   Python runtime environment
-   Required dependencies
-   Flask application
-   Model scoring logic
-   Trained ML model

The container launches the application automatically.

------------------------------------------------------------------------

# Dockerfile

The Dockerfile performs the following steps:

1.  Uses a Python base image
2.  Sets the working directory
3.  Copies application files
4.  Installs dependencies
5.  Exposes the Flask port
6.  Starts the application using

```{=html}
<!-- -->
```
    python app.py

Example Dockerfile:

    FROM python:3.11

    WORKDIR /app

    COPY app.py .
    COPY score.py .
    COPY model.pkl .
    COPY requirements.txt .

    RUN pip install --no-cache-dir -r requirements.txt

    EXPOSE 5000

    CMD ["python", "app.py"]

------------------------------------------------------------------------

# Building the Docker Image

Build the Docker image using:

    docker build -t spam-flask .

------------------------------------------------------------------------

# Running the Docker Container

Run the container with port binding:

    docker run -p 5000:5000 spam-flask

The application becomes accessible at:

    http://127.0.0.1:5000

------------------------------------------------------------------------

# Docker Integration Test

A **Docker integration test** is implemented in `test_assignment4.py`.

Function:

    test_docker()

This test performs the following steps:

1.  Builds the Docker image

```{=html}
<!-- -->
```
    docker build -t spam-flask .

2.  Runs the Docker container

```{=html}
<!-- -->
```
    docker run -p 5001:5000 spam-flask

3.  Sends a request to the API endpoint

```{=html}
<!-- -->
```
    http://127.0.0.1:5001/score

4.  Validates that the response contains:

-   prediction
-   propensity

5.  Terminates the container.

Example test logic:

    response = requests.post(
        "http://127.0.0.1:5001/score",
        json={"text": "Win money now"}
    )

------------------------------------------------------------------------

# Test Coverage

Coverage is generated using:

    pytest --cov=. --cov-report=term-missing > coverage.txt

Example output:

    Name                  Stmts   Miss  Cover
    -----------------------------------------
    app.py                   18      1    94%
    score.py                 11      0   100%
    test_assignment4.py      48      0   100%
    -----------------------------------------
    TOTAL                    99%

Remark:

The line

    if __name__ == "__main__":

is not executed during pytest because the module is imported rather than
run directly.

------------------------------------------------------------------------

# Continuous Integration

A **Git pre‑commit hook** is used to enforce testing before every
commit.

The hook automatically runs:

    py -m pytest

If any tests fail, the commit is aborted.

Location of hook:

    hooks/pre-commit

Example pre‑commit script:

    #!/bin/sh

    echo "Running tests before commit..."

    py -m pytest

    if [ $? -ne 0 ]; then
        echo "Tests failed. Commit aborted."
        exit 1
    fi

Activate the hook:

    Copy-Item hooks/pre-commit .git/hooks/pre-commit

------------------------------------------------------------------------

# Dependencies

Install dependencies using:

    pip install -r requirements.txt

Main libraries:

-   Flask
-   scikit-learn
-   numpy
-   pytest
-   requests

------------------------------------------------------------------------

# Summary

This assignment demonstrates a **complete machine learning deployment
pipeline** including:

-   Flask API serving
-   Docker containerization
-   Docker integration testing
-   Automated testing with pytest
-   Continuous Integration using Git hooks
-   Test coverage reporting

The project successfully validates that the containerized ML application
works correctly and maintains **high test coverage (≈99%)**.
