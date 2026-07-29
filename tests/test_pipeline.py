"""End-to-end tests for build_preprocessing_pipeline.

These cover the properties the transformer unit tests cannot see: step ordering, what
survives to the model, and that fitting on one frame and transforming another (the CV
fold case, and the train/test case) stays consistent.
"""

import numpy as np
import pytest
from preprocessing import OPTIONAL_FEATURES, build_preprocessing_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

ALL_FEATURES = sorted(OPTIONAL_FEATURES)
# what a config actually ships, per configs/gradient_boosting.yaml
SHIPPED_FEATURES = ["male_x_3rdclass", "is_master", "family_group"]


# ## Config validation


def test_unknown_feature_name_raises():
    with pytest.raises(ValueError, match="unknown feature"):
        build_preprocessing_pipeline(["male_x_3rdclass", "not_a_feature"])


def test_unknown_feature_error_lists_the_valid_names():
    with pytest.raises(ValueError) as excinfo:
        build_preprocessing_pipeline(["typo"])
    for name in OPTIONAL_FEATURES:
        assert name in str(excinfo.value)


def test_no_features_is_valid():
    # configs/minimal.yaml and configs/gender_baseline.yaml both pass an empty list
    assert isinstance(build_preprocessing_pipeline([]), Pipeline)


# ## Step ordering
#
# The ordering constraints are invisible at runtime until something raises a KeyError
# deep inside a CV fold, so pin them down here.


def test_title_is_added_before_name_is_dropped():
    names = [name for name, _ in build_preprocessing_pipeline().steps]
    assert names.index("add_title") < names.index("drop_columns")


def test_optional_features_land_after_imputation_and_before_one_hot():
    names = [name for name, _ in build_preprocessing_pipeline(ALL_FEATURES).steps]
    for feature in ALL_FEATURES:
        position = names.index(f"add_{feature}")
        # Age must already be filled -- under10 compares against it
        assert names.index("impute_age") < position
        # and the new columns need to reach one-hot's passthrough
        assert position < names.index("one_hot_encode")


def test_title_is_dropped_before_one_hot():
    # Title is a string column; one-hot's passthrough would hand it straight to the model
    names = [name for name, _ in build_preprocessing_pipeline().steps]
    assert names.index("drop_title") < names.index("one_hot_encode")


def test_is_master_is_added_before_title_is_dropped():
    names = [name for name, _ in build_preprocessing_pipeline(["is_master"]).steps]
    assert names.index("add_is_master") < names.index("drop_title")


# ## Output shape and contents


def test_output_is_entirely_numeric_and_complete(train_df):
    """The contract with the model: no NaNs, no text. LogisticRegression raises on either."""
    out = build_preprocessing_pipeline(SHIPPED_FEATURES).fit_transform(train_df)
    assert not out.isna().any().any()
    assert all(np.issubdtype(dtype, np.number) for dtype in out.dtypes)


def test_raw_columns_do_not_survive(train_df):
    out = build_preprocessing_pipeline(SHIPPED_FEATURES).fit_transform(train_df)
    dropped = {"PassengerId", "Name", "Ticket", "Cabin", "Title", "Pclass", "Embarked"}
    assert not dropped & set(out.columns)


def test_row_count_and_order_survive_the_pipeline(train_df):
    """Every step must be row-preserving: the labels are held alongside, matched by
    position, so a dropped or reordered row would silently mislabel the data."""
    out = build_preprocessing_pipeline(ALL_FEATURES).fit_transform(train_df)
    assert len(out) == len(train_df)
    # Age is untouched wherever it was already present, so it doubles as a row fingerprint
    present = train_df["Age"].notna().to_numpy()
    assert out["Age"].to_numpy()[present] == pytest.approx(
        train_df["Age"].to_numpy()[present]
    )


def test_one_hot_expands_pclass_and_embarked(train_df):
    out = build_preprocessing_pipeline().fit_transform(train_df)
    assert {"Pclass_1", "Pclass_2", "Pclass_3"} <= set(out.columns)
    assert {"Embarked_C", "Embarked_Q", "Embarked_S"} <= set(out.columns)


@pytest.mark.parametrize(
    "feature, columns",
    [
        ("male_x_3rdclass", ["Male_and_3rdClass"]),
        ("is_master", ["IsMaster"]),
        ("under10", ["Under10"]),
        ("family_group", ["FamilySmall", "FamilyLarge"]),
    ],
)
def test_optional_feature_columns_appear_only_when_requested(
    train_df, feature, columns
):
    """The whole point of minimal.yaml is that features: [] really means none of them."""
    with_it = build_preprocessing_pipeline([feature]).fit_transform(train_df)
    without_it = build_preprocessing_pipeline([]).fit_transform(train_df)
    assert set(columns) <= set(with_it.columns)
    assert not set(columns) & set(without_it.columns)


def test_always_on_cleaning_ignores_the_feature_list(train_df):
    # HasCabin and the raw SibSp/Parch are cleaning, not optional features -- minimal.yaml
    # is only meaningful if these are present in both runs
    out = build_preprocessing_pipeline([]).fit_transform(train_df)
    assert {"HasCabin", "SibSp", "Parch", "Sex", "Age", "Fare"} <= set(out.columns)


# ## Fitting on one frame, transforming another


def test_fold_transform_uses_only_the_fitted_statistics(train_df):
    """The leak guard at pipeline level: a held-out fold's own ages must not inform the
    values used to fill its gaps."""
    pipeline = build_preprocessing_pipeline(SHIPPED_FEATURES)
    fitted_on = train_df.iloc[:400]
    pipeline.fit(fitted_on)

    held_out = train_df.iloc[400:].copy()
    filled = pipeline.transform(held_out)

    # every fill value must be one the first 400 rows could have produced
    imputer = pipeline.named_steps["impute_age"]
    allowed = set(imputer.group_stats_["_fill_value"].dropna()) | {imputer.global_stat_}
    was_missing = held_out["Age"].isna().to_numpy()
    assert was_missing.any(), "fixture no longer exercises imputation"
    assert set(filled["Age"].to_numpy()[was_missing]) <= allowed


def test_train_fitted_pipeline_transforms_the_kaggle_test_set(train_df, test_df):
    """What train_submission.py does. test.csv has a missing Fare that train.csv does not,
    so this is the only path where impute_fare does any work."""
    pipeline = build_preprocessing_pipeline(SHIPPED_FEATURES)
    train_out = pipeline.fit_transform(train_df)
    test_out = pipeline.transform(test_df)

    # identical columns in identical order, or the model sees features shuffled under it
    assert list(test_out.columns) == list(train_out.columns)
    assert not test_out.isna().any().any()
    assert len(test_out) == len(test_df)


def test_unseen_category_does_not_add_a_column(train_df):
    """handle_unknown="ignore" means an Embarked value absent from training encodes as
    all-zeros rather than raising or widening the frame."""
    pipeline = build_preprocessing_pipeline()
    train_out = pipeline.fit_transform(train_df)

    odd = train_df.iloc[:5].copy()
    odd["Embarked"] = "Z"
    odd_out = pipeline.transform(odd)

    assert list(odd_out.columns) == list(train_out.columns)
    embarked_cols = [c for c in odd_out.columns if c.startswith("Embarked_")]
    assert (odd_out[embarked_cols].to_numpy() == 0).all()


# ## Smoke test against a real estimator


def test_pipeline_trains_inside_cross_validate(train_df, train_labels):
    """Catches anything that only breaks once sklearn clones and refits per fold."""
    estimator = Pipeline(
        [
            ("preprocessing", build_preprocessing_pipeline(SHIPPED_FEATURES)),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
    scores = cross_val_score(
        estimator, train_df, train_labels, cv=3, scoring="accuracy"
    )
    # a loose floor: anything above the ~0.62 "everyone dies" rate means the pipeline
    # delivered usable features. This is not a performance assertion -- see wandb for those
    assert scores.mean() > 0.70
