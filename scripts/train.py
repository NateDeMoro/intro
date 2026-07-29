import argparse
import os
from pathlib import Path

import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ml_utils import log_cv_results
from preprocessing import build_preprocessing_pipeline

MODELS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": HistGradientBoostingClassifier,
    # at max_depth 1 this is the "all women survive" rule, learned rather than hardcoded:
    # Sex is the strongest single split in the data, so the stump finds it on its own
    "decision_stump": DecisionTreeClassifier,
}

# anchor paths to the project root rather than the shell's cwd, so the command
# works from any directory inside the project
ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"

# wandb otherwise drops its local run logs wherever you happened to invoke from
os.environ.setdefault("WANDB_DIR", str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        # listing the configs as choices gives us --help text and a helpful
        # error on a typo for free
        choices=sorted(p.stem for p in CONFIG_DIR.glob("*.yaml")),
        help="Name of a config in configs/ (without the .yaml extension)",
    )
    args = parser.parse_args()

    with open(CONFIG_DIR / f"{args.config}.yaml") as f:
        config = yaml.safe_load(f)

    df = pd.read_csv(ROOT / "data" / "train.csv")
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    model_cls = MODELS[config["model"]]
    params = config.get("params", {})
    features = config.get("features", [])
    search = config.get("search", {})

    steps = [("preprocessing", build_preprocessing_pipeline(features, config.get("keep")))]
    if config.get("scale"):
        # only meaningful for models that care about feature magnitude (linear ones);
        # fitted per fold by cross_validate, so it can't leak
        steps.append(("scale", StandardScaler()))
    steps.append(("model", model_cls(**params)))
    pipeline = Pipeline(steps)

    estimator = pipeline
    if search:
        # "model__C" addresses the C argument of the "model" step inside the pipeline
        grid = {f"model__{key}": values for key, values in search.items()}
        # this inner split picks the parameters; the outer split below scores the result.
        # Scoring on the same folds used to pick would be optimistic, since the winner
        # was chosen to suit those exact folds
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        n_iter = config.get("n_iter")
        if n_iter:
            # sample n_iter combinations instead of all of them -- cheaper, and better
            # per unit of compute once the grid has more than a couple of dimensions
            estimator = RandomizedSearchCV(
                pipeline,
                grid,
                n_iter=n_iter,
                cv=inner_cv,
                scoring="accuracy",
                n_jobs=-1,
                random_state=42,
            )
        else:
            estimator = GridSearchCV(
                pipeline, grid, cv=inner_cv, scoring="accuracy", n_jobs=-1
            )

    # each fold refits the whole pipeline -- including preprocessing -- on that
    # fold's training portion alone, so imputation/encoding never sees that
    # fold's held-out rows.
    # repeated because a single 5-way cut of 891 rows is worth about +/- 1pp of pure
    # shuffle luck -- enough to have made gradient_boosting look 0.9pp better than
    # logistic_regression when 10 repeats put them level. 50 fits, seconds each; with a
    # `search:` block it multiplies by the sweep size, so drop n_repeats when sweeping
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    cv_results = cross_validate(
        estimator, X, y, cv=cv, scoring=["accuracy", "precision", "recall", "f1"]
    )

    best_params = None
    if search:
        # refit over all the data to report the parameters you'd actually ship
        estimator.fit(X, y)
        best_params = {
            key.removeprefix("model__"): value
            for key, value in estimator.best_params_.items()
        }
        ranked = sorted(
            zip(
                estimator.cv_results_["params"],
                estimator.cv_results_["mean_test_score"],
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )
        print(f"top 5 of {len(ranked)} combinations tried:")
        for tried, score in ranked[:5]:
            label = ", ".join(
                f"{k.removeprefix('model__')}={v}" for k, v in sorted(tried.items())
            )
            print(f"  {score:.3f}  {label}")

    log_cv_results(
        cv_results,
        project="titanic",
        config={
            "model": config["model"],
            **params,
            "features": features,
            "scale": bool(config.get("scale")),
            "search": search or None,
            "best_params": best_params,
        },
        # the config's filename is the run name, so the two can never drift
        run_name=args.config,
    )


if __name__ == "__main__":
    main()
