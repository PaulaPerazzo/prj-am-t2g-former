import os
import json
import random
import time
import warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import optuna
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

SEED = 42


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs():
    os.makedirs("optuna_results", exist_ok=True)


def log_line(dataset_name: str, text: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{dataset_name}] {text}\n"

    ds_log = f"optuna_results/{dataset_name}_catboost_log.txt"
    all_log = "optuna_results/all_catboost_log.txt"

    for path in [ds_log, all_log]:
        with open(path, "a") as f:
            f.write(line)


def load_dataset(dataset_name: str):
    train = pd.read_csv(f"train_datasets/{dataset_name}.csv")
    return train


def prepare_features(df: pd.DataFrame):
    """Prepara features numéricas e categóricas."""
    y = LabelEncoder().fit_transform(df["current_target_class"].values)

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    num_cols = (
        df.select_dtypes(include=np.number)
        .drop(columns=["current_target_class"])
        .columns.tolist()
    )

    df_copy = df.copy()

    # CatBoost lida nativamente com categóricas, mas precisamos converter para int
    cat_indices = []
    if len(cat_cols) > 0:
        for c in cat_cols:
            df_copy[c] = LabelEncoder().fit_transform(df_copy[c].astype(str))
        cat_indices = list(range(len(num_cols), len(num_cols) + len(cat_cols)))

    X_num = df_copy[num_cols].values.astype(np.float32)

    if len(cat_cols) > 0:
        X_cat = df_copy[cat_cols].values.astype(np.int32)
        X = np.hstack([X_num, X_cat])
    else:
        X = X_num

    return X, y, cat_indices, num_cols, cat_cols


def objective(trial, X, y, cat_indices):
    """Função objetivo do Optuna para CatBoost."""

    params = {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
        "rsm": trial.suggest_float("rsm", 0.5, 1.0),  # random subspace method
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),

        # Parâmetros fixos
        "random_seed": SEED,
        "verbose": False,
        "thread_count": 1,
        "task_type": "CPU",
    }

    # 5-Fold Cross-Validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_scores = []

    for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        # Normalizar apenas numéricas (antes das categóricas)
        n_num = X.shape[1] - len(cat_indices)
        scaler = StandardScaler()
        X_train[:, :n_num] = scaler.fit_transform(X_train[:, :n_num])
        X_valid[:, :n_num] = scaler.transform(X_valid[:, :n_num])

        model = CatBoostClassifier(**params)

        if len(cat_indices) > 0:
            model.fit(
                X_train, y_train,
                cat_features=cat_indices,
                eval_set=(X_valid, y_valid),
                early_stopping_rounds=50,
                verbose=False
            )
        else:
            model.fit(
                X_train, y_train,
                eval_set=(X_valid, y_valid),
                early_stopping_rounds=50,
                verbose=False
            )

        preds = model.predict(X_valid)
        acc = accuracy_score(y_valid, preds)
        fold_scores.append(acc)

        # Pruning
        trial.report(acc, fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_scores))


def run_optuna_for_dataset(dataset_name: str, n_trials: int = 50):
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    log_line(dataset_name, f"=== Início tuning CatBoost ({n_trials} trials) ===")

    df = load_dataset(dataset_name)
    X, y, cat_indices, num_cols, cat_cols = prepare_features(df)

    print(f"  Amostras: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"  Numéricas: {len(num_cols)}, Categóricas: {len(cat_cols)}")
    print(f"  Classes: {len(np.unique(y))}")

    start_time = time.time()

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    )

    study.optimize(
        lambda trial: objective(trial, X.copy(), y, cat_indices),
        n_trials=n_trials,
        show_progress_bar=True
    )

    tune_time = time.time() - start_time

    print(f"\n  Melhor Accuracy: {study.best_value:.4f}")
    print(f"  Tempo de tuning: {tune_time:.1f}s")
    print(f"  Melhores parâmetros: {study.best_params}")

    # Salvar resultados
    results = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "tune_time": tune_time,
        "n_trials": n_trials,
        "dataset": dataset_name,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "n_classes": len(np.unique(y)),
    }

    results_path = f"optuna_results/{dataset_name}_catboost_best.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    log_line(dataset_name, f"Tuning finalizado | ACC={study.best_value:.4f} | tempo={tune_time:.1f}s")
    log_line(dataset_name, f"Resultados salvos em {results_path}")

    return study.best_params, tune_time


if __name__ == "__main__":
    seed_everything(SEED)
    ensure_dirs()

    # Lista completa dos 30 datasets
    dataset_list = [
        # Datasets balanceados
        'credit-approval', 'dresses-sales', 'mfeat-morphological', 'vehicle',
        'banknote-authentication', 'analcatdata_dmft', 'MiceProtein', 'cylinder-bands',
        'semeion', 'cnae-9', 'vowel',
        # Datasets desbalanceados
        'breast-w', 'eucalyptus', 'wdbc', 'pc4', 'credit-g', 'cmc',
        'blood-transfusion-service-center', 'pc3', 'car', 'kc2',
        'steel-plates-fault', 'balance-scale', 'pc1', 'tic-tac-toe',
        'analcatdata_authorship', 'climate-model-simulation-crashes',
        'qsar-biodeg', 'diabetes', 'ilpd',
    ]

    total_start = time.time()

    for ds in dataset_list:
        try:
            run_optuna_for_dataset(ds, n_trials=50)
        except Exception as e:
            print(f"Erro no dataset {ds}: {e}")
            log_line(ds, f"ERRO: {repr(e)}")
            continue

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"Tempo total: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'='*60}")
