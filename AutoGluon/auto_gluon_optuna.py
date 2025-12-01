import os
import shutil
import warnings
import json
import time
import numpy as np
import pandas as pd
import optuna

from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from imblearn.metrics import geometric_mean_score

# -------------------------------------------------------------------
# Configurações globais
# -------------------------------------------------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
SEED = 42

# -------------------------------------------------------------------
# 1. metric_report — métrica alvo por dataset
# -------------------------------------------------------------------
metric_report = {
    'breast-w': 'g-mean',
    'eucalyptus': 'g-mean',
    'wdbc': 'g-mean',
    'pc4': 'g-mean',
    'credit-g': 'g-mean',
    'cmc': 'g-mean',
    'blood-transfusion-service-center': 'g-mean',
    'pc3': 'g-mean',
    'car': 'g-mean',
    'kc2': 'g-mean',
    'steel-plates-fault': 'g-mean',
    'balance-scale': 'g-mean',
    'pc1': 'g-mean',
    'tic-tac-toe': 'g-mean',
    'analcatdata_authorship': 'g-mean',
    'climate-model-simulation-crashes': 'g-mean',
    'qsar-biodeg': 'g-mean',
    'diabetes': 'g-mean',
    'ilpd': 'g-mean',

    'credit-approval': 'accuracy',
    'dresses-sales': 'accuracy',
    'mfeat-morphological': 'accuracy',
    'vehicle': 'accuracy',
    'banknote-authentication': 'accuracy',
    'analcatdata_dmft': 'accuracy',
    'MiceProtein': 'accuracy',
    'cylinder-bands': 'accuracy',
    'semeion': 'accuracy',
    'cnae-9': 'accuracy',
    'vowel': 'accuracy'
}

# Espaços de busca de modelos (para Optuna)
MODEL_TYPE_OPTIONS = {
    "gbm_only": ["GBM"],
    "gbm_cat": ["GBM", "CAT"],
    "gbm_xgb": ["GBM", "XGB"],
    "tree_ensemble": ["GBM", "RF", "XT"],
    "gbm_cat_xgb": ["GBM", "CAT", "XGB"],
    "all_trees": ["GBM", "CAT", "XGB", "RF", "XT"],
}

# Presets mais seguros (excluímos 'extreme_quality' e 'best_quality' para evitar TabPFN/TabICL e configs muito pesadas)
VALID_PRESETS = [
    "medium_quality",
    "good_quality",
    "high_quality",
]

# -------------------------------------------------------------------
# 2. Utilitários de diretório
# -------------------------------------------------------------------
def ensure_clean_dir(path: str):
    """Remove o diretório se existir e recria vazio."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

# -------------------------------------------------------------------
# 3. Split dataset 70/30 estratificado
# -------------------------------------------------------------------
def split_dataset(file_path, test_size=0.3):
    df = pd.read_csv(file_path)
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=SEED,
        stratify=df.iloc[:, -1]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

# -------------------------------------------------------------------
# 4. Métricas para um conjunto (train ou test)
# -------------------------------------------------------------------
def compute_fold_metrics(predictor: TabularPredictor, df: pd.DataFrame, target: str):
    y = df[target]
    y_pred = predictor.predict(df)
    y_proba = predictor.predict_proba(df)

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "gmean": geometric_mean_score(y, y_pred),
        "cross_entropy": log_loss(y, y_proba)
    }

    classes = np.unique(y)
    if len(classes) == 2:
        # binário: pega a coluna da classe positiva
        metrics["auc_ovo"] = roc_auc_score(y, y_proba.iloc[:, 1])
    else:
        # multiclasse
        metrics["auc_ovo"] = roc_auc_score(
            y, y_proba, multi_class="ovo", average="macro"
        )

    return metrics

# -------------------------------------------------------------------
# 5. Cross-validation externa 10-fold com AutoGluon
# -------------------------------------------------------------------
def cross_val_autogluon(
    train_df: pd.DataFrame,
    target: str,
    preset: str,
    model_types: list[str],
    metric: str,
    n_splits: int = 10,
    trial = 'final'
):
    """
    Faz CV externa 10-fold com AutoGluon, retornando as MÉDIAS das métricas.
    """
    # limpa diretório base da CV (por trial)
    ensure_clean_dir(f"{trial}_autogluon_cv_temp")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    all_metrics = {
        "accuracy": [],
        "gmean": [],
        "cross_entropy": [],
        "auc_ovo": []
    }

    y_full = train_df[target]

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(train_df, y_full)):
        df_train = train_df.iloc[train_idx].reset_index(drop=True)
        df_val   = train_df.iloc[val_idx].reset_index(drop=True)

        fold_path = os.path.join(f"{trial}_autogluon_cv_temp", f"fold_{fold_idx}")
        ensure_clean_dir(fold_path)

        predictor = TabularPredictor(
            label=target,
            eval_metric="balanced_accuracy" if metric == "g-mean" else "accuracy",
            path=fold_path
        )

        # Aqui é o pulo do gato: limitamos os modelos via included_model_types
        predictor.fit(
            df_train,
            presets=preset,
            included_model_types=model_types,
            # hyperparameters=None  # deixamos o AG usar defaults para esses tipos
            verbosity=0,

            num_bag_folds=0,
            num_stack_levels=0,
            holdout_frac=0,
            time_limit=480
        )

        fold_m = compute_fold_metrics(predictor, df_val, target)
        for k in all_metrics:
            all_metrics[k].append(fold_m[k])

    # média das métricas
    return {k: float(np.mean(v)) for k, v in all_metrics.items()}

# -------------------------------------------------------------------
# 6. Optuna HPO em cima do AutoGluon
# -------------------------------------------------------------------
def run_optuna_autogluon(train_df, target, metric, n_trials=10, name='random'):

    def objective(trial):
        preset = trial.suggest_categorical("preset", VALID_PRESETS)
        model_set_key = trial.suggest_categorical("model_set", list(MODEL_TYPE_OPTIONS.keys()))
        model_types = MODEL_TYPE_OPTIONS[model_set_key]

        cv_metrics = cross_val_autogluon(
            train_df=train_df,
            target=target,
            preset=preset,
            model_types=model_types,
            metric=metric,
            n_splits=10,
            trial=f"{name}_{trial.number}"
        )

        # métrica alvo p/ Optuna
        return cv_metrics["accuracy"] if metric == "accuracy" else cv_metrics["gmean"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)

    best_params = study.best_params
    return best_params  # contém "preset" e "model_set"

# -------------------------------------------------------------------
# 7. Loop principal sobre datasets
# -------------------------------------------------------------------
results = {}

for filename in os.listdir("datasets"):
    if not filename.endswith(".csv"): continue

    print(f"\nAnalyzing {filename}")
    dataset_name = filename.replace(".csv", "")

    if dataset_name not in metric_report:
        print(f"  [WARN] Dataset {dataset_name} não está em metric_report, pulando.")
        continue

    train_df, test_df = split_dataset(os.path.join("datasets", filename))
    target = "current_target_class"
    metric = metric_report[dataset_name]

    # --------------------------
    # OPTUNA TUNING (preset + conjunto de modelos)
    # --------------------------
    start = time.time()
    best_params = run_optuna_autogluon(train_df, target, metric, n_trials=10, name=dataset_name)
    tune_time = time.time() - start

    best_preset = best_params["preset"]
    best_model_set_key = best_params["model_set"]
    best_model_types = MODEL_TYPE_OPTIONS[best_model_set_key]

    # --------------------------
    # MÉTRICAS DE TREINO = MÉDIAS DA CV COM MELHORES PARAMS
    # --------------------------
    train_metrics = cross_val_autogluon(
        train_df=train_df,
        target=target,
        preset=best_preset,
        model_types=best_model_types,
        metric=metric,
        n_splits=10,
        trial=f"{dataset_name}_result"
    )

    # --------------------------
    # TREINO FINAL PARA TESTE
    # --------------------------
    final_path = f"autogluon_best_{dataset_name}"
    ensure_clean_dir(final_path)

    predictor = TabularPredictor(
        label=target,
        eval_metric="balanced_accuracy" if metric == "g-mean" else "accuracy",
        path=final_path
    )

    train_start = time.time()
    predictor.fit(
        train_df,
        presets=best_preset,
        included_model_types=best_model_types,
        verbosity=0,

        num_bag_folds=0,
        num_stack_levels=0,
        holdout_frac=0,
        time_limit=480
    )
    train_time = time.time() - train_start

    pred_start = time.time()
    predictor.predict(test_df)
    pred_time = time.time() - pred_start

    test_metrics = compute_fold_metrics(predictor, test_df, target)

    results[dataset_name] = {
        "best_params": best_params,
        "train_metrics": train_metrics,   # médias da CV
        "test_metrics": test_metrics,     # métricas no teste 30%
        "total_time": tune_time + train_time + pred_time
    }

# -------------------------------------------------------------------
# 8. Salva resultados
# -------------------------------------------------------------------
with open("autogluon_optuna_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nDONE! Results saved to autogluon_optuna_results.json")
