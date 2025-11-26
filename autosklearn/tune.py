import os
import json
import random
import numpy as np
import pandas as pd
import time
import optuna
import warnings

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["DISABLE_JAEGER_TRACING"] = "1"
os.environ["AS_DISABLE_FILE_OUTPUT"] = "1"

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from autosklearn.classification import AutoSklearnClassifier
from sklearn.metrics import accuracy_score


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs():
    os.makedirs("optuna_results", exist_ok=True)
    os.makedirs("optuna_trials", exist_ok=True)


def load_dataset(dataset_name):
    return pd.read_csv(f"train_datasets/{dataset_name}.csv")


def prepare_features(df):
    y = LabelEncoder().fit_transform(df["current_target_class"].values)

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    num_cols = df.select_dtypes(include=np.number).drop(columns=["current_target_class"]).columns.tolist()

    if len(cat_cols) > 0:
        for c in cat_cols:
            df[c] = LabelEncoder().fit_transform(df[c].astype(str))
        X_cat = df[cat_cols].values
    else:
        X_cat = None

    X_num = df[num_cols].values.astype(np.float32)
    return X_num, X_cat, y


def gmean_score(y_true, y_pred, eps=1e-9):
    classes = np.unique(y_true)
    recalls = [
        ((y_true == c) & (y_pred == c)).sum() / ((y_true == c).sum() + eps)
        for c in classes
    ]
    return np.prod(recalls) ** (1 / len(recalls))


def objective(trial, dataset_name):

    df = load_dataset(dataset_name)
    X_num, X_cat, y = prepare_features(df)

    scaler = StandardScaler()
    X_num = scaler.fit_transform(X_num)

    X = np.hstack([X_num, X_cat]) if X_cat is not None else X_num

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Hiperparâmetros
    time_limit = trial.suggest_int("time_left_for_this_task", 60, 300, step=60)
    run_limit = trial.suggest_int("per_run_time_limit", 15, 60, step=15)
    ensemble = trial.suggest_int("ensemble_size", 10, 50, step=10)

    automl = AutoSklearnClassifier(
        seed=42,
        time_left_for_this_task=time_limit,
        per_run_time_limit=run_limit,
        ensemble_kwargs={"ensemble_size": ensemble},
        memory_limit=4096,
        n_jobs=1,
        resampling_strategy="holdout",
        resampling_strategy_arguments={"train_size": 0.8}
    )

    automl.fit(X_train, y_train)
    preds = automl.predict(X_valid)
    # score = gmean_score(y_valid, preds)
    score = accuracy_score(y_valid, preds)

    # 🔥 SALVAR RESULTADO DO TRIAL IMEDIATAMENTE 🔥
    trial_path = f"optuna_trials/{dataset_name}_trial_{trial.number}.json"
    with open(trial_path, "w") as f:
        json.dump(
            {
                "trial_number": trial.number,
                "params": trial.params,
                "value": score,
                "time": time.time(),
            },
            f,
            indent=4
        )

    return score


def run_optuna(dataset_name, n_trials=20):
    print(f"\n🔍 Dataset: {dataset_name}")

    study = optuna.create_study(direction="maximize")

    study.optimize(lambda t: objective(t, dataset_name), n_trials=n_trials)

    # Salvar melhor resultado ao final
    with open(f"optuna_results/{dataset_name}_best.json", "w") as f:
        json.dump(
            {
                "best_value": study.best_value,
                "best_params": study.best_params
            },
            f,
            indent=4
        )

    print("✔ Finalizado:", dataset_name)
    print(" → Best:", study.best_value)
    print(" → Params:", study.best_params)


if __name__ == "__main__":
    seed_everything(42)
    ensure_dirs()

    dataset_list = [
        'credit-approval', 'dresses-sales', 'mfeat-morphological', 'vehicle',
        'banknote-authentication', 'analcatdata_dmft', 'MiceProtein', 'cylinder-bands',
        'semeion', 'cnae-9', 'vowel'
    ]

    total = time.time()

    for ds in dataset_list:
        run_optuna(ds, n_trials=20)

    print("\nTempo total:", time.time() - total)
