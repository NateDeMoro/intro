# ---
# jupyter:
#   jupytext:
#     formats: ipynb,../scripts//py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Preprocessing Pipeline
#
# Cleaning and feature logic only — no EDA. Paired to `scripts/preprocessing.py` via Jupytext, which `train.py` and `explore.ipynb` both import from.

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ## Transformers

# drop columns with no predictive value
class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns)


# map Sex from text to 0/1
class SexEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["Sex"] = X["Sex"].map({"female": 0, "male": 1})
        return X


# fill target_col with a per-group stat; fit() sees only the rows it's given,
# so inside a CV fold it never learns from that fold's held-out rows
class GroupStatImputer(BaseEstimator, TransformerMixin):
    def __init__(self, group_cols, target_col, stat):
        self.group_cols = group_cols
        self.target_col = target_col
        self.stat = stat

    def fit(self, X, y=None):
        self.group_stats_ = (
            X.groupby(self.group_cols)[self.target_col]
            .agg(self.stat)
            .rename("_fill_value")
            .reset_index()
        )
        return self

    def transform(self, X):
        merged = X.merge(self.group_stats_, on=self.group_cols, how="left")
        merged[self.target_col] = merged[self.target_col].fillna(merged["_fill_value"])
        return merged.drop(columns=["_fill_value"])


# flag the one Sex x Pclass interaction that tested significant in explore.ipynb
class InteractionFeatureAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["Male_and_3rdClass"] = ((X["Sex"] == 1) & (X["Pclass"] == 3)).astype(int)
        return X


# flag passengers under 10 -- the age histogram in explore.ipynb shows a survival spike there
class ChildFlagAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["Under10"] = (X["Age"] < 10).astype(int)
        return X


# ## Pipeline

# engineered features a config can switch on by name via its `features` list
OPTIONAL_FEATURES = {
    "male_x_3rdclass": InteractionFeatureAdder,
    "under10": ChildFlagAdder,
}


# raw DataFrame -> model-ready features; safe to hand straight to cross_validate()
def build_preprocessing_pipeline(features=()):
    unknown = set(features) - set(OPTIONAL_FEATURES)
    if unknown:
        raise ValueError(
            f"unknown feature(s) {sorted(unknown)}; valid: {sorted(OPTIONAL_FEATURES)}"
        )

    one_hot = ColumnTransformer(
        transformers=[
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["Pclass", "Embarked"],
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    one_hot.set_output(transform="pandas")

    steps = [
        (
            "drop_columns",
            ColumnDropper(["PassengerId", "Name", "Ticket", "Cabin"]),
        ),
        ("encode_sex", SexEncoder()),
        (
            "impute_age",
            GroupStatImputer(
                group_cols=["Pclass", "Sex"], target_col="Age", stat="median"
            ),
        ),
        (
            "impute_embarked",
            GroupStatImputer(
                group_cols=["Pclass"],
                target_col="Embarked",
                stat=lambda s: s.mode().iloc[0],
            ),
        ),
    ]
    # after imputation so Age is already filled, before one-hot so the new columns pass through
    steps += [(f"add_{name}", OPTIONAL_FEATURES[name]()) for name in features]
    steps.append(("one_hot_encode", one_hot))

    return Pipeline(steps)
