import os
import json
import time
import warnings
import random
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from autosklearn.classification import AutoSklearnClassifier

warnings.filterwarnings("ignore")

# -----------------------------
# Preparação
# -----------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["AS_DISABLE_FILE_OUTPUT"] = "1"


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def load_dataset(ds):
    df = pd.read_csv(f"train_datasets/{ds}.csv")
    
    return df


def prepare_features(df):
    y = LabelEncoder().fit_transform(df["current_target_class"])

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    num_cols = df.select_dtypes(include=np.number).drop(columns=["current_target_class"]).columns.tolist()

    df2 = df.copy()

    if cat_cols:
        for c in cat_cols:
            df2[c] = LabelEncoder().fit_transform(df2[c].astype(str))
    
        X_cat = df2[cat_cols].values
    
    else:
        X_cat = None

    X_num = df2[num_cols].astype(np.float32).values
    
    return X_num, X_cat, y


def gmean_score(y_true, y_pred, eps=1e-9):
    recalls = []
    
    for c in np.unique(y_true):
        tp = ((y_true == c) & (y_pred == c)).sum()
        fn = ((y_true == c) & (y_pred != c)).sum()
        recalls.append(tp / (tp + fn + eps))
    
    return np.prod(recalls) ** (1 / len(recalls))


def evaluate_model(model, X_valid, y_valid):
    preds = model.predict(X_valid)
    
    return gmean_score(y_valid, preds)


def run_halving(ds):
    print(f"\n🔍 Dataset: {ds}")

    df = load_dataset(ds)
    X_num, X_cat, y = prepare_features(df)

    # Escalonamento
    scaler = StandardScaler()
    X_num = scaler.fit_transform(X_num)

    if X_cat is not None:
        X = np.hstack([X_num, X_cat])
    
    else:
        X = X_num

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Search Space
    param_grid = {
        "time_left_for_this_task": [60, 120, 180, 240],
        "per_run_time_limit": [15, 30, 45],
        "ensemble_kwargs": [
            {"ensemble_size": 10},
            {"ensemble_size": 20},
            {"ensemble_size": 40},
        ],
    }

    base_model = AutoSklearnClassifier(
        seed=42,
        memory_limit=4096,
        n_jobs=1,
        resampling_strategy="holdout",
        resampling_strategy_arguments={"train_size": 0.8},
    )

    # Halving Search
    search = HalvingGridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        factor=2,
        resource="n_samples",
        max_resources=X_train.shape[0],
        scoring=None,  # gmean manual
        cv=3,
        verbose=1
    )

    start = time.time()
    search.fit(X_train, y_train)
    duration = time.time() - start

    best_cfg = search.best_params_
    print("Best params:", best_cfg)

    # Avaliação final no conjunto holdout
    best_model = search.best_estimator_
    g = evaluate_model(best_model, X_valid, y_valid)

    os.makedirs("halving_results", exist_ok=True)

    with open(f"halving_results/{ds}.json", "w") as f:
        json.dump(
            {
                "dataset": ds,
                "best_params": best_cfg,
                "gmean_valid": g,
                "time_seconds": duration,
            },
            f,
            indent=4
        )

    print(f"Finalizado {ds}")
    print(f" G-Mean: {g}")
    print(f" Tempo: {duration}s")


if __name__ == "__main__":
    seed_everything(42)

    dataset_list = [
        'breast-w','eucalyptus', 'wdbc', 'pc4', 'credit-g', 'cmc',
        'blood-transfusion-service-center', 'pc3', 'car', 'kc2',
        'steel-plates-fault', 'balance-scale', 'pc1', 'tic-tac-toe',
        'analcatdata_authorship', 'climate-model-simulation-crashes',
        'qsar-biodeg', 'diabetes', 'ilpd'
    ]

    for ds in dataset_list:
        run_halving(ds)
