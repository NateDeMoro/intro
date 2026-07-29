"""Validation for every YAML in configs/.

A broken config currently fails at the worst possible moment: after argparse accepts
it, partway through a 50-fit repeated-CV run, having already opened a wandb run. These
tests catch a typo'd hyperparameter or feature name in under a second instead.
"""

import pytest
import yaml
from preprocessing import OPTIONAL_FEATURES, build_preprocessing_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from train import CONFIG_DIR, MODELS

CONFIG_PATHS = sorted(CONFIG_DIR.glob("*.yaml"))
CONFIG_NAMES = [path.stem for path in CONFIG_PATHS]

KNOWN_KEYS = {"model", "features", "params", "scale", "search", "n_iter"}


@pytest.fixture(params=CONFIG_PATHS, ids=CONFIG_NAMES)
def config(request):
    """Every config in configs/, one test run each -- so a new file is covered the
    moment it is added, without touching this file."""
    with open(request.param) as f:
        return yaml.safe_load(f)


def test_configs_exist():
    assert CONFIG_PATHS, "no configs found -- has configs/ moved?"


def test_config_uses_only_known_keys(config):
    # train.py silently ignores anything it does not recognise, so a misspelled
    # "feature:" or "param:" would just not take effect
    assert set(config) <= KNOWN_KEYS


def test_config_names_a_registered_model(config):
    assert config["model"] in MODELS


def test_config_features_are_all_known(config):
    assert set(config.get("features", [])) <= set(OPTIONAL_FEATURES)


def test_config_params_are_accepted_by_the_model(config):
    """sklearn estimators take explicit keyword arguments, so instantiating with the
    config's params is enough to catch a misspelled hyperparameter."""
    MODELS[config["model"]](**config.get("params", {}))


def test_config_search_keys_are_real_hyperparameters(config):
    search = config.get("search") or {}
    valid = MODELS[config["model"]]().get_params()
    assert set(search) <= set(valid)


def test_n_iter_only_appears_with_a_search_grid(config):
    # without a `search:` block train.py never reads n_iter, so its presence alone
    # means someone expected a sweep that will not happen
    if "n_iter" in config:
        assert config.get("search"), "n_iter set but no search grid to sample from"


def test_config_builds_a_pipeline_that_fits(config, train_df, train_labels):
    """Assembles the config exactly the way train.py does and fits it once. Catches a
    feature/model combination that only fails on contact with the data."""
    steps = [
        ("preprocessing", build_preprocessing_pipeline(config.get("features", [])))
    ]
    if config.get("scale"):
        steps.append(("scale", StandardScaler()))
    steps.append(("model", MODELS[config["model"]](**config.get("params", {}))))

    fitted = Pipeline(steps).fit(train_df, train_labels)
    predictions = fitted.predict(train_df)
    assert len(predictions) == len(train_df)
    assert set(predictions) <= {0, 1}


def test_scaling_is_only_requested_for_models_that_need_it(config):
    """Trees split on thresholds, so a StandardScaler in front of one is dead weight and
    a sign the config was copied from the logistic regression one."""
    tree_models = {"random_forest", "gradient_boosting", "decision_stump"}
    if config["model"] in tree_models:
        assert not config.get("scale")
