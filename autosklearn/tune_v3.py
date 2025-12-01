import os
import json
import random
import time
import warnings
import logging

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from autosklearn.classification import AutoSklearnClassifier

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["DISABLE_JAEGER_TRACING"] = "1"
os.environ["AS_DISABLE_FILE_OUTPUT"] = "1"

logging.getLogger("smac").setLevel(logging.ERROR)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs():
    os.makedirs("optuna_results", exist_ok=True)


def log_line(dataset_name: str, text: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{dataset_name}] {text}\n"

    ds_log_path = os.path.join("optuna_results", f"{dataset_name}_log.txt")
    all_log_path = os.path.join("optuna_results", "all_datasets_log.txt")

    for path in [ds_log_path, all_log_path]:
        with open(path, "a") as f:
            f.write(line)


def load_dataset(dataset_name: str) -> pd.DataFrame:
    return pd.read_csv(f"train_datasets/{dataset_name}.csv")


def prepare_features(train_df: pd.DataFrame):
    # Target
    y = LabelEncoder().fit_transform(train_df["current_target_class"].values)

    cat_cols = train_df.select_dtypes(include="object").columns.tolist()
    num_cols = (
        train_df.select_dtypes(include=np.number)
        .drop(columns=["current_target_class"])
        .columns
        .tolist()
    )

    df = train_df.copy()

    if len(cat_cols) > 0:
        for c in cat_cols:
            df[c] = LabelEncoder().fit_transform(df[c].astype(str))
        X_cat = df[cat_cols].values
    else:
        X_cat = None

    X_num_raw = df[num_cols].values.astype(np.float32)

    return X_num_raw, X_cat, y


def gmean_score(y_true, y_pred, eps=1e-9):
    classes = np.unique(y_true)
    recalls = []

    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        recalls.append(tp / (tp + fn + eps))

    return float(np.prod(recalls) ** (1.0 / len(recalls)))


# 2. RANDOM SEARCH
def sample_hyperparams():
    cfg = {}

    # time per trial
    cfg["time_left_for_this_task"] = random.choice([20, 30, 40])

    cfg["per_run_time_limit"] = random.choice([5, 10])

    cfg["ensemble_size"] = random.choice([1, 5])

    cfg["train_size"] = random.choice([0.7, 0.8])

    cfg["memory_limit"] = random.choice([2048, 3072, 4096])  # MB

    return cfg


def run_random_search_for_dataset(dataset_name: str, n_trials: int = 10):
    print(f"dataset: {dataset_name}")
    log_line(dataset_name, f"=== random search ({n_trials} trials) ===")

    df = load_dataset(dataset_name)
    X_num, X_cat, y = prepare_features(df)

    scaler = StandardScaler()
    X_num = scaler.fit_transform(X_num)

    if X_cat is not None:
        X = np.hstack([X_num, X_cat])
    else:
        X = X_num

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    best_score = -np.inf
    best_params = None

    results_path = os.path.join("optuna_results", f"{dataset_name}_random_search.json")

    for trial_idx in range(1, n_trials + 1):
        hp = sample_hyperparams()

        msg = (
            f"Trial {trial_idx}/{n_trials} | "
            f"time_left={hp['time_left_for_this_task']}s, "
            f"per_run={hp['per_run_time_limit']}s, "
            f"ensemble_size={hp['ensemble_size']}, "
            f"train_size={hp['train_size']}, "
            f"memory={hp['memory_limit']}MB"
        )
        
        print(f"[{dataset_name}] {msg}")
        log_line(dataset_name, msg)

        start_trial = time.time()

        try:
            automl = AutoSklearnClassifier(
                seed=42,
                time_left_for_this_task=hp["time_left_for_this_task"],
                per_run_time_limit=hp["per_run_time_limit"],
                ensemble_kwargs={"ensemble_size": hp["ensemble_size"]},
                memory_limit=hp["memory_limit"],
                n_jobs=1,
                resampling_strategy="holdout",
                resampling_strategy_arguments={"train_size": hp["train_size"]},
            )

            automl.fit(X_train, y_train)
            preds = automl.predict(X_valid)

            score = gmean_score(y_valid, preds)

            elapsed = time.time() - start_trial
            log_line(dataset_name, f"Trial {trial_idx} finalizado | g-mean={score:.4f} | tempo={elapsed:.1f}s")

            if score > best_score:
                best_score = score
                best_params = hp

                with open(results_path, "w") as f:
                    json.dump(
                        {
                            "best_value": best_score,
                            "best_params": best_params,
                            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "n_trials_done": trial_idx,
                        },
                        f,
                        indent=4,
                    )

                log_line(dataset_name, f"best_score={score:.4f} salvo em {results_path}")

        except Exception as e:
            elapsed = time.time() - start_trial
            err_msg = f"Trial {trial_idx} FALHOU após {elapsed:.1f}s | erro: {repr(e)}"
            print(f"[{dataset_name}] {err_msg}")
            log_line(dataset_name, err_msg)
            continue

    log_line(dataset_name, f"=== random search | best_score={best_score:.4f} ===")
    print(f" Dataset {dataset_name} finalizado | best_score={best_score:.4f}")


if __name__ == "__main__":
    seed_everything(42)
    ensure_dirs()

    dataset_list = [
        "breast-w", "eucalyptus", "wdbc", "pc4", "credit-g", "cmc",
        "blood-transfusion-service-center", "pc3", "car", "kc2",
        "steel-plates-fault", "balance-scale", "pc1", "tic-tac-toe",
        "analcatdata_authorship", "climate-model-simulation-crashes",
        "qsar-biodeg", "diabetes", "ilpd",
    ]

    total_start = time.time()

    for ds in dataset_list:
        run_random_search_for_dataset(ds, n_trials=10)

    total_time = time.time() - total_start
    print(f"Tempo total (todos os datasets): {total_time:.1f}s")
