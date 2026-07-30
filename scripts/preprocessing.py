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


# keep only the named columns, in the order given. the mirror of ColumnDropper, for when the
# point is to state what the model may see rather than what it may not -- an sklearn estimator
# consumes whatever array it is handed, so restricting it has to happen here
class ColumnKeeper(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        # the trailing underscore is not decoration: this is the last step of the
        # preprocessing pipeline, and sklearn decides a Pipeline is fitted by asking its
        # final step. a transformer that sets nothing makes the whole pipeline look unfitted
        self.columns_ = list(self.columns)
        return self

    def transform(self, X):
        missing = [column for column in self.columns if column not in X.columns]
        if missing:
            raise KeyError(
                f"keep names {missing}, which the pipeline does not produce; "
                f"available: {sorted(X.columns)}"
            )
        return X[list(self.columns)]


# pull the honorific out of Name ("Braund, Mr. Owen Harris" -> "Mr"). Title is a working
# column, not a feature: impute_age groups on it and drop_title removes it before one-hot
class TitleAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        title = (
            X["Name"]
            .str.extract(r",\s*([^\.]+)\.", expand=False)
            .str.strip()
            .replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
        )
        # Dr/Rev/Col/Lady/... are ~2% of rows between them, too thin to fit individually
        X["Title"] = title.where(title.isin(["Mr", "Mrs", "Miss", "Master"]), "Rare")
        return X


# map Sex from text to 0/1
class SexEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["Sex"] = X["Sex"].map({"female": 0, "male": 1})
        return X


# replace Cabin with a 0/1 "was a cabin recorded" flag -- the value is 77% missing, but
# explore.ipynb shows the missingness itself carries signal (70% vs 29% survival, and it
# holds inside every Pclass, p=0.001 controlling for Pclass, Sex and Fare)
class CabinFlagEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["HasCabin"] = X["Cabin"].notna().astype(int)
        return X.drop(columns=["Cabin"])


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
        # a group can be all-NaN, or absent from the fold we fit on -- both leave the merge
        # below with nothing to fill from, so keep an ungrouped stat as the backstop
        self.global_stat_ = X[self.target_col].agg(self.stat)
        return self

    def transform(self, X):
        merged = X.merge(self.group_stats_, on=self.group_cols, how="left")
        merged[self.target_col] = merged[self.target_col].fillna(merged["_fill_value"])
        merged[self.target_col] = merged[self.target_col].fillna(self.global_stat_)
        return merged.drop(columns=["_fill_value"])


# flag the one Sex x Pclass interaction that tested significant in explore.ipynb
class InteractionFeatureAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["Male_and_3rdClass"] = ((X["Sex"] == 1) & (X["Pclass"] == 3)).astype(int)
        return X


# the one exception to "women survive": 3rd-class women in a family of 5+ died at 89%, where
# 3rd-class women overall are an exact coin flip. explore_3rd_class_women.ipynb has it at +21 rows
# against the gender baseline. it has to be an explicit three-way flag -- a greedy tree splits
# female on Pclass before it ever reaches family size, and that split lands on a 50/50 leaf.
# note the 1st/2nd-class restriction is load-bearing: all 6 large-family women up there survived
class Female3rdLargeFamilyAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        family_size = X["SibSp"] + X["Parch"] + 1
        X["Female_3rd_LargeFamily"] = (
            (X["Sex"] == 0) & (X["Pclass"] == 3) & (family_size >= 5)
        ).astype(int)
        return X


# 3rd-class women aged 18-30 travelling with a sibling or spouse. by raw age this is 12 rows at
# 17% survival, but note what the pipeline actually hands it: optional features run after
# impute_age, and the Pclass 3 / female median is 21.5 -- inside the band. So every missing-age
# 3rd-class woman with SibSp > 0 is swept in too, making it 29 rows at 35%. Same net rows either
# way (+8 vs +9), different feature. Move this step before impute_age if you want the raw version
class Female3rdYoungSiblingAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["Female_3rd_YoungSibling"] = (
            (X["Sex"] == 0)
            & (X["Pclass"] == 3)
            & X["Age"].between(18, 30)
            & (X["SibSp"] > 0)
        ).astype(int)
        return X


# flag boys -- "Master" was the period's honorific for a child male. explore_remaining.ipynb
# has this beating Under10 head to head: it reads the name rather than the age, so it still
# catches a boy whose Age was missing and got imputed to the adult median
class MasterFlagAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["IsMaster"] = (X["Title"] == "Master").astype(int)
        return X


# the mirror of Female_3rd_LargeFamily, and the stronger half of it: a boy travelling in a family
# of 2-4 survived 22 times out of 22 in train.csv, across all three classes (3/3, 9/9, 10/10),
# while boys in families of 5+ went down 17 times out of 18. small families got their children
# into boats; large ones drowned together. worth +22 rows where a plain IsMaster flag is worth +6,
# so the family condition is the whole point. see the scan in explore_1st_class_men.ipynb's sibling
# analysis -- and note gradient_boosting, which has IsMaster and family_group, finds 21 of the 22
# on its own, which is independent evidence the pattern is not an artifact of searching for it
class BoySmallFamilyAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        family_size = X["SibSp"] + X["Parch"] + 1
        X["Boy_SmallFamily"] = (
            (X["Title"] == "Master") & family_size.between(2, 4)
        ).astype(int)
        return X


# flag passengers under 10 -- the age histogram in explore.ipynb shows a survival spike there
class ChildFlagAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["Under10"] = (X["Age"] < 10).astype(int)
        return X


# bucket family size (SibSp + Parch + self) into alone / 2-4 / 5+ -- survival by family size
# is an inverted U in explore.ipynb, so a linear term cancels out where the buckets don't.
# alone is the reference level: both flags are 0 for it
class FamilyGroupAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        family_size = X["SibSp"] + X["Parch"] + 1
        X["FamilySmall"] = family_size.between(2, 4).astype(int)
        X["FamilyLarge"] = (family_size >= 5).astype(int)
        return X


# ## Pipeline

# engineered features a config can switch on by name via its `features` list
OPTIONAL_FEATURES = {
    "male_x_3rdclass": InteractionFeatureAdder,
    "female_3rd_largefamily": Female3rdLargeFamilyAdder,
    "female_3rd_young_sibling": Female3rdYoungSiblingAdder,
    "is_master": MasterFlagAdder,
    "boy_small_family": BoySmallFamilyAdder,
    "under10": ChildFlagAdder,
    "family_group": FamilyGroupAdder,
}


# raw DataFrame -> model-ready features; safe to hand straight to cross_validate().
# `keep` restricts the output to those columns, which is the only way to hold a model to a
# subset -- use it to build a deliberately small model rather than to prune a large one
def build_preprocessing_pipeline(features=(), keep=None):
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
        # before drop_columns, which is where Name goes
        ("add_title", TitleAdder()),
        (
            "drop_columns",
            ColumnDropper(["PassengerId", "Name", "Ticket"]),
        ),
        ("encode_sex", SexEncoder()),
        ("encode_cabin", CabinFlagEncoder()),
        (
            "impute_age",
            # grouping on Title instead of Sex ages the missing-age boys correctly (~4 rather
            # than ~26), but explore_remaining.ipynb shows it costs accuracy once is_master is
            # on -- the imputed age then duplicates the flag and the trees lose a split
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
        (
            # train.csv has no gaps here, but test.csv has one -- and a NaN reaching
            # LogisticRegression is an exception rather than a bad prediction
            "impute_fare",
            GroupStatImputer(
                group_cols=["Pclass"], target_col="Fare", stat="median"
            ),
        ),
    ]
    # after imputation so Age is already filled, before one-hot so the new columns pass through
    steps += [(f"add_{name}", OPTIONAL_FEATURES[name]()) for name in features]
    # Title has done its work by here -- as a string column it would survive one-hot's
    # passthrough and hand the model text
    steps.append(("drop_title", ColumnDropper(["Title"])))
    steps.append(("one_hot_encode", one_hot))
    # last, so `keep` names post-one-hot columns ("Pclass_3", not "Pclass")
    if keep:
        steps.append(("keep_columns", ColumnKeeper(keep)))

    return Pipeline(steps)
