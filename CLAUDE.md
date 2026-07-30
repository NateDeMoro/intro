# intro

Titanic survival classifier. sklearn pipelines, YAML-driven configs, wandb for run history.

## Layout

| Path | What it is | Open when... |
| --- | --- | --- |
| `scripts/preprocessing.py` | Custom transformers + `build_preprocessing_pipeline(features)`. Raw DataFrame in, model-ready features out. | Changing cleaning, encoding, or adding an engineered feature. |
| `scripts/train.py` | CLI entrypoint (`train <config>`). Loads a config, nests search inside repeated CV (5 folds x 10 repeats), logs to wandb. | Changing how models are selected, scored, or logged. |
| `scripts/train_submission.py` | Fits a config on all of `train.csv`, predicts `test.csv`, writes `submissions/<config>-<timestamp>.csv`. No metrics — nothing is held out. | Producing a file to upload to Kaggle. |
| `scripts/ml_utils.py` | wandb logging helpers (`log_cv_results`, `evaluate_and_log`) + a pre-pipeline `fill_with_group_stat` left over from the notebook era. | Changing what gets logged per run. |
| `configs/*.yaml` | One per model family: `model`, `features`, optional `scale`, `params`, `search`/`n_iter`. Edited in place; wandb keeps the history. | Tuning hyperparameters or toggling features. |
| `configs/gender_baseline.yaml` | The control: a depth-1 tree, which is exactly "all women survive". Read every other score as a delta from this. | Judging whether a change is worth anything. |
| `configs/gender_3.yaml` | The current best line of work: baseline + two exceptions (`female_3rd_largefamily`, `boy_small_family`), 0.835 CV. | Extending the baseline-plus-exceptions model. |
| `configs/gender_plus_family_lr.yaml` | Same rule as `gender_plus_family`, as a logistic regression on two columns via `keep:`. Exact, no side effects — prefer this one. | Expressing "baseline plus named exceptions" precisely. |
| `configs/gender_plus_family.yaml` | The baseline plus one carve-out (`female_3rd_largefamily`). Needs `max_depth: 3` — see the config's comment. | Building up from the baseline one feature at a time. |
| `configs/minimal.yaml` | `gradient_boosting` params with every optional feature off. The gap to `gradient_boosting.yaml` is what the engineered features are worth. | Re-measuring the feature set's value. |
| `notebooks/preprocessing.ipynb` | Jupytext pair of `scripts/preprocessing.py`. | Never edit alone — see Jupytext below. |
| `notebooks/explore.ipynb` | EDA: Sex, Pclass, Age, Fare, SibSp/Parch, Cabin. Evidence for every engineered feature. | Justifying or questioning a feature. |
| `notebooks/explore_remaining.ipynb` | EDA part two: Name/Title, Ticket, Embarked, fare-per-person, interactions. Ends in a CV table of candidates. | Considering a feature not already in the pipeline — several are tested and rejected here. |
| `notebooks/explore_1st_class_men.ipynb` | The other big pocket. Concludes it is a genuine coin flip: no subgroup clears break-even, and `gradient_boosting` beats the baseline by 3 rows out of 122. | Before spending effort on 1st-class men — the answer is don't. |
| `notebooks/explore_3rd_class_women.ipynb` | The biggest error pocket, dissected. `Embarked=S` and "male on ticket" are both proxies for family size; a second pass over the 117-row residual finds no per-passenger signal left. | Before adding any feature aimed at 3rd-class women. |
| `notebooks/explore_men.ipynb` | Male subgroups above break-even, with a calibration panel: the top-ranked candidate predicted +12.8 test rows and delivered 0. Read that before trusting any subgroup scan. | Tempted by a subgroup that looks great in training. |
| `notebooks/error_analysis.ipynb` | Out-of-fold errors of `gender_baseline` by Sex x Pclass pocket, with the net-rows-available maths. | Deciding what to build next — start here, not from a feature idea. |
| `data/train.csv`, `data/test.csv` | Kaggle Titanic data. | — |
| `tests/conftest.py` | Fixtures: a hand-built 8-row `raw` frame covering every transformer branch, plus session-scoped `train_df`/`train_labels`/`test_df` off the real CSVs. | Adding a test that needs a new edge case — extend `raw` rather than reading a CSV. |
| `tests/test_transformers.py` | Each transformer in isolation, with expected values read off the `raw` fixture. Includes the `GroupStatImputer` leak and row-order guards. | Changing a transformer's logic. |
| `tests/test_pipeline.py` | `build_preprocessing_pipeline` end to end: step ordering, numeric/complete output, optional-feature gating, fit-on-train/transform-on-test parity. | Changing the pipeline's shape or step order. |
| `.github/workflows/ci.yml` | GitHub Actions: `uv sync --locked --group dev` then `uv run pytest`, on push to main and every PR. `WANDB_MODE: offline` so nothing can block on a login. | Adding a CI step, or when a run goes red. |
| `tests/test_configs.py` | Parametrized over every `configs/*.yaml` — unknown keys, unregistered model, bad feature name, hyperparameter that the estimator won't accept, and a real fit. | Adding a config or a model to `MODELS`. |

## Conventions

- **Jupytext pair**: `scripts/preprocessing.py` <-> `notebooks/preprocessing.ipynb`. Edit the `.py`, then run `.venv/bin/jupytext --sync notebooks/preprocessing.ipynb`. Never let the two drift.
- **Leak-free by construction**: every transformer learns in `fit()` only, so `cross_validate` refits preprocessing per fold. Don't compute statistics over the full frame.
- **Optional features** are gated by name through `OPTIONAL_FEATURES`; always-on cleaning lives in the fixed `steps` list. A config referencing an unknown name raises.
- **`Title` is a working column, not a feature**: `add_title` creates it before `drop_columns` (where `Name` goes), `impute_age` and `is_master` consume it, `drop_title` removes it before one-hot. Anything reading `Name` or `Ticket` must be inserted before `drop_columns`.
- **EDA before engineering**: a new feature needs a cell in `explore.ipynb` backing it. Note the cost: every feature here was chosen by eyeballing all 891 training rows, so CV overstates them — the pipeline's ~+4.5pp over a plain "women survive" rule did not survive contact with the leaderboard.
- **`keep:` restricts what the model sees**: an sklearn estimator consumes every column handed to it, so a config's `keep:` list appends a `ColumnKeeper` after one-hot. Names are post-one-hot (`Pclass_3`, not `Pclass`). Use it to build a deliberately small model, not to prune a large one.
- **A stateless transformer cannot be the last pipeline step**: sklearn asks the final step whether a `Pipeline` is fitted, so a transformer that sets no trailing-underscore attribute makes `predict()` raise `NotFittedError` after a successful `fit`. `ColumnKeeper` sets `columns_` for exactly this reason.
- **Greedy trees miss interactions**: a tree splits on whichever feature gives the biggest immediate gain, so `Pclass_3` beats `female_3rd_largefamily` inside the female branch even though that split lands on a 50/50 leaf. An interaction the data supports may still need an explicit feature *and* enough depth — check `export_text` rather than assuming the model found it.
- **One split is not a measurement**: 891 rows means a single 5-fold cut carries ~±1pp of shuffle luck. Compare variants on repeated CV, never on one `random_state`.
- Run Python via `.venv/bin/python` from the project root (paths in notebooks are root-relative).
- **Tests**: `.venv/bin/python -m pytest` from the project root (~3s, no wandb, no network). `pythonpath = ["scripts"]` in `pyproject.toml` means tests import the working copy, not the editable install. A preprocessing bug here is silent — a wrong regex or flipped encoding yields a valid frame and a slightly worse score, never an exception — so assert expected values by hand, never against the code's own output.
- `tests/test_configs.py` is parametrized over `configs/*.yaml`, so a new config is covered the moment it is added. A new entry in `MODELS` needs no test change either.
- **CI runs what is committed, not what is in your working tree.** `uv sync --locked` also means a `pyproject.toml` change pushed without its regenerated `uv.lock` fails the run rather than silently re-resolving — commit both together.

## Gotchas

- `train.py` writes a wandb run on every invocation — don't run it as a smoke test. Use `cross_val_score` directly instead.
- `train.csv` has no missing `Fare` but `test.csv` does, so `impute_fare` is a no-op during CV and only earns its keep at submission time.
- Don't regroup `impute_age` on `Title`. It looks right (missing-age boys get ~4 instead of ~26) but costs accuracy once `is_master` is on — the imputed age duplicates the flag (gb 0.845 -> 0.834). Decomposition table in `explore_remaining.ipynb`.
- `wandb/` is gitignored local cache; the runs live on wandb.ai.
- Re-saving a notebook with `nbformat` can reflow a cell's `source` from a string to a line list — a no-op diff, not a content change.
