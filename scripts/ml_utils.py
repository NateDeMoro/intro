import wandb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def fill_with_group_stat(train_df, other_df, group_cols, target_col, stat):
    """Compute `stat` (e.g. "median", or a lambda for mode) of target_col per group,
    using train_df only, then fill missing values in both train_df and other_df with
    those group values. Fitting on train only avoids leaking test data into the value
    used to fill train's own gaps.
    """
    group_stats = (
        train_df.groupby(group_cols)[target_col]
        .agg(stat)
        .rename("_fill_value")
        .reset_index()
    )

    def apply_fill(df_):
        merged = df_.merge(group_stats, on=group_cols, how="left")
        merged[target_col] = merged[target_col].fillna(merged["_fill_value"])
        return merged.drop(columns=["_fill_value"])

    return apply_fill(train_df), apply_fill(other_df)


def evaluate_and_log(model, X_test, y_test, project, config, run_name=None):
    """Evaluate a fitted model on X_test/y_test, print metrics, and log the run to wandb."""
    wandb.init(project=project, config=config, name=run_name)

    y_pred = model.predict(X_test)

    metrics = {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    print(f"Test accuracy:  {metrics['test_accuracy']:.3f}")
    print(f"Precision:      {metrics['precision']:.3f}")
    print(f"Recall:         {metrics['recall']:.3f}")
    print(f"F1 score:       {metrics['f1_score']:.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    wandb.log(metrics)
    wandb.finish()

    return metrics


def log_cv_results(cv_results, project, config, run_name=None):
    """Print and log the mean +/- std of each cross-validation metric to wandb.
    cv_results is whatever sklearn's cross_validate() returns.
    """
    wandb.init(project=project, config=config, name=run_name)

    summary = {}
    for key, values in cv_results.items():
        if not key.startswith("test_"):
            continue
        metric_name = key.removeprefix("test_")
        mean, std = values.mean(), values.std()
        summary[f"{metric_name}_mean"] = mean
        summary[f"{metric_name}_std"] = std
        print(f"{metric_name:10s}: {mean:.3f} +/- {std:.3f}")

    wandb.log(summary)
    wandb.finish()

    return summary
