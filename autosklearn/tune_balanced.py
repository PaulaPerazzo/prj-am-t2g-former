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
from sklearn.metrics import accuracy_score
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


# 1. FUNÇÕES AUXILIARES
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs():
    os.makedirs("optuna_results", exist_ok=True)


def log_line(dataset_name: str, text: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{dataset_name}] {text}\n"

    ds_log = f"optuna_results/{dataset_name}_log.txt"
    all_log = "optuna_results/all_datasets_log.txt"

    for path in [ds_log, all_log]:
        with open(path, "a") as f:
            f.write(line)


def load_dataset(dataset_name: str) -> pd.DataFrame:
    return pd.read_csv(f"train_datasets/{dataset_name}.csv")


def prepare_features(df: pd.DataFrame):
    y = LabelEncoder().fit_transform(df["current_target_class"].values)

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    num_cols = (
        df.select_dtypes(include=np.number)
        .drop(columns=["current_target_class"])
        .columns.tolist()
    )

    df2 = df.copy()

    if len(cat_cols) > 0:
        for c in cat_cols:
            df2[c] = LabelEncoder().fit_transform(df2[c].astype(str))

        X_cat = df2[cat_cols].values
    
    else:
        X_cat = None

    X_num = df2[num_cols].values.astype(np.float32)

    return X_num, X_cat, y


# 2. RANDOM SEARCH DE HYPERPARAMETROS (métrica = ACC)
def sample_hyperparams():
    return {
        "time_left_for_this_task": random.choice([20, 30, 40]),
        "per_run_time_limit": random.choice([5, 10]),
        "ensemble_size": random.choice([1, 5]),
        "train_size": random.choice([0.7, 0.8]),
        "memory_limit": random.choice([2048, 3072, 4096]),
    }


def run_random_search_for_dataset(dataset_name: str, n_trials: int = 10):
    print(f"\n🔍 Dataset atual: {dataset_name}")
    log_line(dataset_name, f"=== Início random search ACC ({n_trials} trials) ===")

    df = load_dataset(dataset_name)
    X_num, X_cat, y = prepare_features(df)

    scaler = StandardScaler()
    X_num = scaler.fit_transform(X_num)

    X = np.hstack([X_num, X_cat]) if X_cat is not None else X_num

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    best_score = -np.inf
    best_params = None

    results_path = f"optuna_results/{dataset_name}_random_search.json"

    for t in range(1, n_trials + 1):
        hp = sample_hyperparams()

        msg = (
            f"Trial {t}/{n_trials} | "
            f"time_left={hp['time_left_for_this_task']}s, "
            f"per_run={hp['per_run_time_limit']}s, "
            f"ensemble={hp['ensemble_size']}, "
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

            score = accuracy_score(y_valid, preds)
            elapsed = time.time() - start_trial

            log_line(dataset_name, f"Trial {t} finalizado | ACC={score:.4f} | tempo={elapsed:.1f}s")

            if score > best_score:
                best_score = score
                best_params = hp

                with open(results_path, "w") as f:
                    json.dump(
                        {
                            "best_value": best_score,
                            "best_params": best_params,
                            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "n_trials_done": t,
                        },
                        f,
                        indent=4,
                    )

                log_line(dataset_name, f"★ Novo best_score={score:.4f} salvo!")

        except Exception as e:
            elapsed = time.time() - start_trial
        
            log_line(dataset_name, f"Trial {t} FALHOU após {elapsed:.1f}s | erro: {repr(e)}")
            print(f"[{dataset_name}] Trial {t} falhou: {e}")
        
            continue

    log_line(dataset_name, f"=== Fim random search | BEST_ACC={best_score:.4f} ===")
    print(f"Dataset {dataset_name} finalizado | BEST_ACC={best_score:.4f}")


# 3. MAIN
if __name__ == "__main__":
    seed_everything(42)
    ensure_dirs()

    dataset_list = [
        'credit-approval', 'dresses-sales', 'mfeat-morphological', 'vehicle',
        'banknote-authentication', 'analcatdata_dmft', 'MiceProtein', 'cylinder-bands',
        'semeion', 'cnae-9', 'vowel'
    ]

    total_start = time.time()

    for ds in dataset_list:
        run_random_search_for_dataset(ds, n_trials=10)

    total_time = time.time() - total_start

    print(f"Tempo total (todos os datasets): {total_time:.1f}s")
