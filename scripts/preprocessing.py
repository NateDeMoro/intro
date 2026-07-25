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
# This notebook is paired with `preprocessing.py` via Jupytext — edit either one, then run `uv run jupytext --sync preprocessing.ipynb` to push the change to the other. `train.py` and `explore.ipynb` both import from `preprocessing.py`, so any change made here is picked up automatically the next time `train.py` runs.
#
# This notebook only holds cleaning/feature logic (Transformer classes) — no EDA, no plots. `explore.ipynb` stays separate for that.

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ## ColumnDropper
#
# Drops a fixed list of columns. Nothing to learn from data, so `fit()` is a no-op — it just has to exist to satisfy sklearn's Transformer interface.

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns)


# ## SexEncoder
#
# Maps Sex from "female"/"male" text to 0/1. Deterministic, so `fit()` is again a no-op.

class SexEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["Sex"] = X["Sex"].map({"female": 0, "male": 1})
        return X


# ## GroupStatImputer
#
# Fills missing values in `target_col` using `stat` (e.g. `"median"`, or a mode lambda) computed per `group_cols`. `fit()` only looks at whatever data it's given — inside a cross-validation fold, that's the fold's training split only — so the same fitted values get applied to that fold's held-out split in `transform()`, avoiding leakage between them.

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


# ## InteractionFeatureAdder
#
# Adds `Male_and_3rdClass`, the one interaction the statsmodels check in `explore.ipynb` found statistically significant. Deterministic given Sex/Pclass, so `fit()` is a no-op.

class InteractionFeatureAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["Male_and_3rdClass"] = ((X["Sex"] == 1) & (X["Pclass"] == 3)).astype(int)
        return X


# ## build_preprocessing_pipeline
#
# Everything from a raw Titanic DataFrame (minus `Survived`) to model-ready features, as one `Pipeline`. Every step's `fit()` only uses whatever data it's given — the whole pipeline can be handed directly to `cross_validate()` and each fold refits it correctly on that fold's training data alone.

def build_preprocessing_pipeline():
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

    return Pipeline(
        [
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
            ("add_interaction", InteractionFeatureAdder()),
            ("one_hot_encode", one_hot),
        ]
    )
