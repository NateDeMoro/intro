"""Fit a config on all of train.csv and write Kaggle-ready predictions for test.csv.

Use when you want something to upload rather than a score: every labelled row goes into the fit,
so there is nothing held out to measure and this prints no metrics. Scoring lives in train.py.
"""

import argparse
from datetime import datetime

import pandas as pd
import yaml
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing import build_preprocessing_pipeline
from train import CONFIG_DIR, MODELS, ROOT

SUBMISSION_DIR = ROOT / "submissions"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        choices=sorted(p.stem for p in CONFIG_DIR.glob("*.yaml")),
        help="Name of a config in configs/ (without the .yaml extension)",
    )
    args = parser.parse_args()

    with open(CONFIG_DIR / f"{args.config}.yaml") as f:
        config = yaml.safe_load(f)

    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")
    X = train.drop(columns=["Survived"])
    y = train["Survived"]

    steps = [("preprocessing", build_preprocessing_pipeline(config.get("features", [])))]
    if config.get("scale"):
        steps.append(("scale", StandardScaler()))
    steps.append(("model", MODELS[config["model"]](**config.get("params", {}))))
    estimator = Pipeline(steps)

    search = config.get("search", {})
    if search:
        # the same inner split train.py picks parameters with. Its outer scoring split has no
        # counterpart here -- holding rows back would only make the shipped model worse
        grid = {f"model__{key}": values for key, values in search.items()}
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        n_iter = config.get("n_iter")
        if n_iter:
            estimator = RandomizedSearchCV(
                estimator,
                grid,
                n_iter=n_iter,
                cv=inner_cv,
                scoring="accuracy",
                n_jobs=-1,
                random_state=42,
            )
        else:
            estimator = GridSearchCV(
                estimator, grid, cv=inner_cv, scoring="accuracy", n_jobs=-1
            )

    estimator.fit(X, y)

    if search:
        print("chosen parameters:")
        for key, value in sorted(estimator.best_params_.items()):
            print(f"  {key.removeprefix('model__')}={value}")

    # PassengerId is read off the raw frame -- the pipeline drops it before the model sees it
    submission = pd.DataFrame(
        {"PassengerId": test["PassengerId"], "Survived": estimator.predict(test).astype(int)}
    )

    SUBMISSION_DIR.mkdir(exist_ok=True)
    path = SUBMISSION_DIR / f"{args.config}-{datetime.now():%Y%m%d-%H%M%S}.csv"
    submission.to_csv(path, index=False)

    # no accuracy to report; the nearest sanity check is that the predicted survival rate
    # lands somewhere near the rate the model was trained on
    print(f"\n{len(submission)} predictions -> {path.relative_to(ROOT)}")
    print(
        f"predicted survival rate {submission['Survived'].mean():.3f} "
        f"(train base rate {y.mean():.3f})"
    )


if __name__ == "__main__":
    main()
