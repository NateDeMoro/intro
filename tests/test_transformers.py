"""Unit tests for each transformer in isolation.

Every case here is one a silent bug could pass: a wrong title regex, a flipped Sex
encoding, or an off-by-one family bucket all produce a valid frame and a slightly
worse CV score, never an exception. Checking the numbers by hand is the only way to
catch them.
"""

import pandas as pd
import pytest
from preprocessing import (
    BoySmallFamilyAdder,
    CabinFlagEncoder,
    ChildFlagAdder,
    ColumnDropper,
    ColumnKeeper,
    FamilyGroupAdder,
    Female3rdLargeFamilyAdder,
    Female3rdYoungSiblingAdder,
    GroupStatImputer,
    InteractionFeatureAdder,
    MasterFlagAdder,
    SexEncoder,
    TitleAdder,
)

# ## ColumnDropper


def test_column_dropper_removes_only_named_columns(raw):
    out = ColumnDropper(["PassengerId", "Name", "Ticket"]).fit_transform(raw)
    assert not {"PassengerId", "Name", "Ticket"} & set(out.columns)
    assert {"Pclass", "Sex", "Age", "Fare"} <= set(out.columns)
    assert len(out) == len(raw)


# ## ColumnKeeper


def test_column_keeper_selects_in_the_order_given(raw):
    out = ColumnKeeper(["Fare", "Pclass"]).fit_transform(raw)
    assert out.columns.tolist() == ["Fare", "Pclass"]
    assert len(out) == len(raw)


def test_column_keeper_reports_itself_as_fitted(raw):
    """ColumnKeeper is the last step of the pipeline when `keep:` is set, and sklearn asks
    the final step whether a Pipeline is fitted. A stateless transformer there makes
    pipeline.predict() raise NotFittedError after a successful fit."""
    from sklearn.utils.validation import check_is_fitted

    check_is_fitted(ColumnKeeper(["Fare"]).fit(raw))


def test_column_keeper_names_the_missing_column(raw):
    # the failure mode this guards against is a typo'd `keep:` entry silently handing the
    # model the wrong columns, so the error has to say which name was wrong
    with pytest.raises(KeyError, match="Embarked_S"):
        ColumnKeeper(["Sex", "Embarked_S"]).fit_transform(raw)


# ## TitleAdder


def test_title_adder_extracts_and_collapses_honorifics(raw):
    out = TitleAdder().fit_transform(raw)
    # Mlle/Ms -> Miss and Mme -> Mrs are the collapses; Dr falls into Rare
    assert out["Title"].tolist() == [
        "Mr",
        "Mrs",
        "Miss",
        "Master",
        "Rare",
        "Mrs",
        "Miss",
        "Mr",
    ]


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Braund, Mr. Owen Harris", "Mr"),
        ("Cumings, Mrs. John Bradley", "Mrs"),
        ("Bonnell, Miss. Elizabeth", "Miss"),
        ("Palsson, Master. Gosta", "Master"),
        # the four spellings that get folded into an existing bucket
        ("Bonnell, Mlle. Elizabeth", "Miss"),
        ("Laroche, Ms. Simonne", "Miss"),
        ("Nasser, Mme. Nicholas", "Mrs"),
        # too thin to fit individually -- all of these become Rare
        ("Minahan, Dr. William", "Rare"),
        ("Byles, Rev. Thomas", "Rare"),
        ("Duff Gordon, Lady. Lucy", "Rare"),
        ("Weir, Col. John", "Rare"),
        # a surname containing a comma still resolves to the honorific after the last one
        ("Rothes, the Countess. of Lucy", "Rare"),
    ],
)
def test_title_adder_single_names(name, expected):
    df = pd.DataFrame({"Name": [name]})
    assert TitleAdder().fit_transform(df)["Title"].iloc[0] == expected


def test_title_adder_does_not_mutate_input(raw):
    before = raw.copy()
    TitleAdder().fit_transform(raw)
    pd.testing.assert_frame_equal(raw, before)


# ## SexEncoder


def test_sex_encoder_maps_female_to_zero(raw):
    out = SexEncoder().fit_transform(raw)
    # the direction matters: InteractionFeatureAdder tests `Sex == 1` for male
    assert out["Sex"].tolist() == [1, 0, 0, 1, 1, 0, 0, 1]


# ## CabinFlagEncoder


def test_cabin_flag_encoder_flags_presence_and_drops_the_column(raw):
    out = CabinFlagEncoder().fit_transform(raw)
    assert "Cabin" not in out.columns
    assert out["HasCabin"].tolist() == [1, 1, 0, 0, 1, 0, 0, 0]


# ## GroupStatImputer


def test_group_stat_imputer_fills_from_the_group_median(raw):
    out = GroupStatImputer(
        group_cols=["Pclass", "Sex"], target_col="Age", stat="median"
    ).fit_transform(raw)
    assert out["Age"].notna().all()
    # row 2 is Pclass 3 / female. The only other 3/female row is index 7? no -- row 7 is
    # male. With no other 3/female age present the group median is NaN, so the global
    # median (of 22, 38, 44, 14, 3, 30) backstops it
    assert out.loc[2, "Age"] == raw["Age"].median()
    # row 3 is Pclass 3 / male; row 7 is the other one, age 30
    assert out.loc[3, "Age"] == 30.0


def test_group_stat_imputer_falls_back_when_the_group_is_absent_at_transform_time():
    """A CV fold can fit on rows that never contain some group. The merge then finds
    nothing to fill from, and the global stat has to cover it."""
    fit_frame = pd.DataFrame({"Pclass": [1, 1], "Age": [40.0, 20.0]})
    imputer = GroupStatImputer(group_cols=["Pclass"], target_col="Age", stat="median")
    imputer.fit(fit_frame)

    # Pclass 3 was never seen during fit
    unseen = pd.DataFrame({"Pclass": [3], "Age": [None]})
    assert (
        imputer.transform(unseen)["Age"].iloc[0] == 30.0
    )  # global median of 40 and 20


def test_group_stat_imputer_learns_only_in_fit():
    """The leak guard. If transform recomputed the statistic from the frame it is given,
    the held-out fold would be filling its own gaps with its own values."""
    fit_frame = pd.DataFrame({"Pclass": [1, 1, 1], "Age": [10.0, 10.0, 10.0]})
    imputer = GroupStatImputer(group_cols=["Pclass"], target_col="Age", stat="median")
    imputer.fit(fit_frame)

    # this frame's own median would be 999; the fill must come from fit_frame instead
    held_out = pd.DataFrame({"Pclass": [1, 1, 1], "Age": [999.0, 999.0, None]})
    assert imputer.transform(held_out)["Age"].iloc[2] == 10.0


def test_group_stat_imputer_preserves_row_order():
    """transform() merges, which resets the index. Order is what keeps rows aligned with
    y inside cross_validate, so it is the property worth pinning down."""
    fit_frame = pd.DataFrame({"Pclass": [1, 2, 3], "Age": [40.0, 30.0, 20.0]})
    imputer = GroupStatImputer(group_cols=["Pclass"], target_col="Age", stat="median")
    imputer.fit(fit_frame)

    scrambled = pd.DataFrame(
        {"Pclass": [3, 1, 2, 3], "Age": [None, 5.0, None, 7.0], "Marker": list("abcd")}
    )
    out = imputer.transform(scrambled)
    assert out["Marker"].tolist() == list("abcd")
    assert out["Age"].tolist() == [20.0, 5.0, 30.0, 7.0]


def test_group_stat_imputer_handles_a_mode_statistic(raw):
    """Embarked is filled with a mode rather than a median, via a lambda."""
    out = GroupStatImputer(
        group_cols=["Pclass"], target_col="Embarked", stat=lambda s: s.mode().iloc[0]
    ).fit_transform(raw)
    assert out["Embarked"].notna().all()
    # row 4 is the missing one, Pclass 2; the other Pclass 2 row embarked at C
    assert out.loc[4, "Embarked"] == "C"


# ## Optional feature transformers


def test_interaction_feature_adder_needs_encoded_sex(raw):
    # Sex must already be 0/1 -- the flag tests `Sex == 1`, which is never true for text
    encoded = SexEncoder().fit_transform(raw)
    out = InteractionFeatureAdder().fit_transform(encoded)
    # only rows 3 and 7 are male and in 3rd class
    assert out["Male_and_3rdClass"].tolist() == [0, 0, 0, 1, 0, 0, 0, 1]


@pytest.mark.parametrize(
    "sex, pclass, sibsp, parch, expected",
    [
        (0, 3, 4, 0, 1),  # 3rd-class woman, family of exactly 5 -- the lower edge
        (0, 3, 8, 2, 1),
        (0, 3, 2, 1, 0),  # family of 4, one short
        (0, 1, 4, 0, 0),  # the class restriction: 1st-class women all survived
        (0, 2, 4, 0, 0),
        (1, 3, 4, 0, 0),  # a man in the same family is not the same passenger
    ],
)
def test_female_3rd_large_family_adder_needs_all_three_conditions(
    sex, pclass, sibsp, parch, expected
):
    # Sex must already be encoded -- the flag tests `Sex == 0`, never true for text
    df = pd.DataFrame(
        {"Sex": [sex], "Pclass": [pclass], "SibSp": [sibsp], "Parch": [parch]}
    )
    out = Female3rdLargeFamilyAdder().fit_transform(df)
    assert out["Female_3rd_LargeFamily"].iloc[0] == expected


@pytest.mark.parametrize(
    "sex, pclass, age, sibsp, expected",
    [
        (0, 3, 18.0, 1, 1),  # lower edge of the band
        (0, 3, 30.0, 1, 1),  # upper edge -- between() is inclusive
        (0, 3, 17.0, 1, 0),
        (0, 3, 31.0, 1, 0),
        (0, 3, 25.0, 0, 0),  # no sibling or spouse aboard
        (0, 1, 25.0, 1, 0),  # 3rd class only
        (1, 3, 25.0, 1, 0),  # women only
    ],
)
def test_female_3rd_young_sibling_adder_band_and_conditions(sex, pclass, age, sibsp, expected):
    df = pd.DataFrame({"Sex": [sex], "Pclass": [pclass], "Age": [age], "SibSp": [sibsp]})
    out = Female3rdYoungSiblingAdder().fit_transform(df)
    assert out["Female_3rd_YoungSibling"].iloc[0] == expected


def test_female_3rd_young_sibling_adder_does_not_fire_on_missing_age():
    """In the pipeline this step runs after impute_age, so a missing age arrives as 21.5 and
    the flag fires. On raw NaN it must not -- pinning that keeps the difference deliberate
    rather than accidental if the step order ever changes."""
    df = pd.DataFrame({"Sex": [0], "Pclass": [3], "Age": [None], "SibSp": [1]})
    assert Female3rdYoungSiblingAdder().fit_transform(df)["Female_3rd_YoungSibling"].iloc[0] == 0


def test_master_flag_adder_reads_title_not_age(raw):
    # row 3 is a Master whose Age is missing -- the point of the flag is that it still
    # fires without ever consulting Age
    titled = TitleAdder().fit_transform(raw)
    out = MasterFlagAdder().fit_transform(titled)
    assert out["IsMaster"].tolist() == [0, 0, 0, 1, 0, 0, 0, 0]
    assert pd.isna(out.loc[3, "Age"])


@pytest.mark.parametrize(
    "title, sibsp, parch, expected",
    [
        ("Master", 1, 0, 1),  # family of 2, lower edge of the bucket
        ("Master", 2, 1, 1),  # family of 4, upper edge
        ("Master", 0, 0, 0),  # a boy alone is not the same passenger
        ("Master", 4, 0, 0),  # family of 5 -- these boys mostly died
        ("Mr", 1, 0, 0),  # an adult man in a small family gains nothing
        ("Miss", 1, 0, 0),
    ],
)
def test_boy_small_family_adder_needs_the_title_and_the_bucket(title, sibsp, parch, expected):
    # reads Title, so it must run after TitleAdder and before drop_title
    df = pd.DataFrame({"Title": [title], "SibSp": [sibsp], "Parch": [parch]})
    out = BoySmallFamilyAdder().fit_transform(df)
    assert out["Boy_SmallFamily"].iloc[0] == expected


def test_boy_small_family_adder_matches_family_group_buckets(raw):
    """The 2-4 bucket has to mean the same thing here as in FamilyGroupAdder, or the two
    features silently disagree about what a small family is."""
    titled = TitleAdder().fit_transform(raw)
    boys = BoySmallFamilyAdder().fit_transform(titled)
    buckets = FamilyGroupAdder().fit_transform(titled)
    is_master = titled["Title"] == "Master"
    assert (boys["Boy_SmallFamily"] == (is_master & buckets["FamilySmall"].eq(1))).all()


def test_child_flag_adder_uses_a_strict_under_ten(raw):
    out = ChildFlagAdder().fit_transform(raw)
    # only row 6 (age 3) qualifies; the missing ages are NaN < 10, which is False
    assert out["Under10"].tolist() == [0, 0, 0, 0, 0, 0, 1, 0]


@pytest.mark.parametrize(
    "sibsp, parch, small, large",
    [
        (0, 0, 0, 0),  # alone -- the reference level, both flags off
        (1, 0, 1, 0),  # family of 2, lower edge of small
        (2, 1, 1, 0),  # family of 4, upper edge of small
        (3, 1, 0, 1),  # family of 5, lower edge of large
        (8, 2, 0, 1),
    ],
)
def test_family_group_adder_bucket_edges(sibsp, parch, small, large):
    df = pd.DataFrame({"SibSp": [sibsp], "Parch": [parch]})
    out = FamilyGroupAdder().fit_transform(df)
    assert out["FamilySmall"].iloc[0] == small
    assert out["FamilyLarge"].iloc[0] == large
    # the buckets are exclusive -- a passenger is alone, small, or large, never two
    assert out["FamilySmall"].iloc[0] + out["FamilyLarge"].iloc[0] <= 1
