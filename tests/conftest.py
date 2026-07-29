"""Fixtures shared by the transformer, pipeline, and config tests.

The synthetic `raw` frame is the workhorse: hand-built so every assertion has an
expected value you can read off the fixture, rather than one derived from the same
code under test. The real-CSV fixtures are only for the end-to-end checks that need
the actual missingness pattern (test.csv's lone missing Fare, train.csv's two
missing Embarked).
"""

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def raw():
    """Eight rows covering every branch the transformers have.

    Deliberate edge cases, by row index:
      0-1  Mr/Mrs, cabin recorded, family of 2 (FamilySmall)
      2    Mlle -> Miss, missing Age (Pclass 3/female group)
      3    Master, missing Age (Pclass 3/male group), family of 7 (FamilyLarge)
      4    Dr -> Rare, alone, missing Embarked
      5    Mme -> Mrs, missing Fare
      6    Ms -> Miss, family of exactly 4 (upper edge of FamilySmall)
      7    Mr in Pclass 3 (the Male_and_3rdClass interaction), family of exactly 5
    """
    return pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4, 5, 6, 7, 8],
            "Pclass": [1, 1, 3, 3, 2, 2, 1, 3],
            "Name": [
                "Braund, Mr. Owen Harris",
                "Cumings, Mrs. John Bradley",
                "Bonnell, Mlle. Elizabeth",
                "Palsson, Master. Gosta Leonard",
                "Minahan, Dr. William Edward",
                "Nasser, Mme. Nicholas",
                "Laroche, Ms. Simonne",
                "Sage, Mr. John George",
            ],
            "Sex": [
                "male",
                "female",
                "female",
                "male",
                "male",
                "female",
                "female",
                "male",
            ],
            "Age": [22.0, 38.0, None, None, 44.0, 14.0, 3.0, 30.0],
            "SibSp": [1, 1, 0, 4, 0, 1, 1, 4],
            "Parch": [0, 0, 0, 2, 0, 0, 2, 0],
            "Ticket": [
                "A/5 21171",
                "PC 17599",
                "113783",
                "349909",
                "19928",
                "2649",
                "SC/Paris 2123",
                "CA. 2343",
            ],
            "Fare": [7.25, 71.28, 26.55, 21.07, 90.0, None, 41.58, 69.55],
            "Cabin": ["C85", "C123", None, None, "B58", None, None, None],
            "Embarked": ["S", "C", "S", "S", None, "C", "C", "S"],
        }
    )


@pytest.fixture(scope="session")
def train_df():
    """The real training data, minus the label -- i.e. exactly what train.py hands the pipeline."""
    return pd.read_csv(ROOT / "data" / "train.csv").drop(columns=["Survived"])


@pytest.fixture(scope="session")
def train_labels():
    """The Survived column, aligned by position with `train_df`."""
    return pd.read_csv(ROOT / "data" / "train.csv")["Survived"]


@pytest.fixture(scope="session")
def test_df():
    """The real Kaggle test set. Unlike train.csv it has a missing Fare, so it is the
    only fixture that exercises impute_fare for real."""
    return pd.read_csv(ROOT / "data" / "test.csv")
