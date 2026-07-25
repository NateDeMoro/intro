import argparse
import os
from pathlib import Path

import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from ml_utils import log_cv_results
from preprocessing import build_preprocessing_pipeline

MODELS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
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
    pipeline = Pipeline(
        [
            ("preprocessing", build_preprocessing_pipeline()),
            ("model", model_cls(**params)),
        ]
    )

    # each fold refits the whole pipeline -- including preprocessing -- on that
    # fold's training portion alone, so imputation/encoding never sees that
    # fold's held-out rows
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        pipeline, X, y, cv=cv, scoring=["accuracy", "precision", "recall", "f1"]
    )

    log_cv_results(
        cv_results,
        project="titanic",
        config={"model": config["model"], **params, "features": config.get("features")},
        # the config's filename is the run name, so the two can never drift
        run_name=args.config,
    )


if __name__ == "__main__":
    main()
