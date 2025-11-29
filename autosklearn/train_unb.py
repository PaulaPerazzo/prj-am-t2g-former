import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from autosklearn.classification import AutoSklearnClassifier

warnings.filterwarnings("ignore")


def ensure_dirs():
    os.makedirs("trained_models_askl", exist_ok=True)
    os.makedirs("results", exist_ok=True)


def load_best_params(dataset_name):
    """hiperparâmetros do Optuna."""
    path = f"optuna_results/{dataset_name}_best.json"
    
    with open(path, "r") as f:
        data = json.load(f)
    
    return data["best_params"], data.get("tune_time", np.nan)


def load_dataset(dataset_name):
    train = pd.read_csv(f"train_datasets/{dataset_name}.csv")
    test  = pd.read_csv(f"test_datasets/{dataset_name}.csv")
    
    return train, test


def preprocess(df):
    df = df.copy()

    # label
    y = LabelEncoder().fit_transform(df["current_target_class"])

    # detecta colunas
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    num_cols = (
        df.select_dtypes(include=np.number)
        .drop(columns=["current_target_class"])
        .columns.tolist()
    )

    # codifica categóricas
    if len(cat_cols) > 0:
        for c in cat_cols:
            df[c] = LabelEncoder().fit_transform(df[c].astype(str))
        
        X_cat = df[cat_cols].values
    
    else:
        X_cat = None

    X_num = df[num_cols].values.astype(np.float32)

    return X_num, X_cat, y


# MÉTRICAS
def gmean_score(y_true, y_pred, eps=1e-9):
    classes = np.unique(y_true)
    recalls = []

    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        recalls.append(tp / (tp + fn + eps))

    return float(np.prod(recalls) ** (1.0 / len(recalls)))


def compute_auc_ovo(y_true, probs):
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    unique_classes = np.unique(y_true)

    if len(unique_classes) < 2:
        return np.nan

    try:
        if len(unique_classes) > 2:
            return roc_auc_score(
                y_true,
                probs[:, unique_classes],
                multi_class="ovo",
                average="macro"
            )

        else:
            return roc_auc_score(y_true, probs[:, 1])

    except:
        return np.nan


def evaluate_split(model, X, y):
    start = time.time()
    probs = model.predict_proba(X)
    tempo_predict = time.time() - start

    probs = np.asarray(probs)
    y = np.asarray(y)

    unique_classes = np.unique(y)
    n_classes = len(unique_classes)

    if probs.shape[1] != n_classes:
        probs_fixed = np.zeros((len(probs), n_classes))

        for i, c in enumerate(model.classes_):
            if c in unique_classes:
                pos = np.where(unique_classes == c)[0][0]
                probs_fixed[:, pos] = probs[:, i]

        probs = probs_fixed

    preds = np.argmax(probs, axis=1)

    return {
        "auc_ovo": compute_auc_ovo(y, probs),
        "mean_acc": accuracy_score(y, preds),
        "g_mean": gmean_score(y, preds),
        "mean_cross_entropy": log_loss(y, probs),
        "tempo_predict": tempo_predict,
    }


# TREINO FINAL
def train_final_model(dataset_name):
    print(f"\n===== Treinando modelo final para {dataset_name} =====")

    best_params, tempo_tune = load_best_params(dataset_name)
    print("Melhores hiperparâmetros:", best_params)

    train_df, test_df = load_dataset(dataset_name)

    X_train_num, X_train_cat, y_train = preprocess(train_df)
    X_test_num,  X_test_cat,  y_test  = preprocess(test_df)

    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train_num)
    X_test_num  = scaler.transform(X_test_num)

    def merge(Xn, Xc):
        return np.hstack([Xn, Xc]) if Xc is not None else Xn

    X_train = merge(X_train_num, X_train_cat)
    X_test  = merge(X_test_num,  X_test_cat)

    # treinar modelo final
    automl = AutoSklearnClassifier(
        seed=42,
        time_left_for_this_task=best_params["time_left_for_this_task"],
        per_run_time_limit=best_params["per_run_time_limit"],
        ensemble_kwargs={"ensemble_size": best_params["ensemble_size"]},
        memory_limit=4096,
        n_jobs=1,
    )

    start_train = time.time()
    automl.fit(X_train, y_train)
    tempo_train = time.time() - start_train

    # salvar modelo
    path = f"trained_models_askl/{dataset_name}_askl.pkl"
    joblib.dump(automl, path)
    print(f"Modelo salvo: {path}")

    # avaliação
    train_metrics = evaluate_split(automl, X_train, y_train)
    test_metrics  = evaluate_split(automl, X_test,  y_test)

    # construir saída
    rows = []

    for split, metrics in [
        ("train", train_metrics),
        ("test",  test_metrics)
    ]:
        rows.append({
            "split": split,
            "nome_modelo": "AutoSklearn",
            "dataset": dataset_name,
            "tempo_tune": tempo_tune,
            "tempo_train": tempo_train,
            "auc_ovo": metrics["auc_ovo"],
            "mean_acc": metrics["mean_acc"],
            "g_mean": metrics["g_mean"],
            "mean_cross_entropy": metrics["mean_cross_entropy"],
            "tempo_predict": metrics["tempo_predict"],
        })

    return rows


# MAIN
if __name__ == "__main__":
    ensure_dirs()

    dataset_list = [
        "breast-w", "eucalyptus", "wdbc", "pc4", "credit-g", "cmc",
        "blood-transfusion-service-center", "pc3", "car", "kc2",
        "steel-plates-fault", "balance-scale", "pc1", "tic-tac-toe",
        "analcatdata_authorship", "climate-model-simulation-crashes",
        "qsar-biodeg", "diabetes", "ilpd",
    ]

    all_rows = []

    for ds in dataset_list:
        rows = train_final_model(ds)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv("results/final_autosklearn_results_unb.csv", index=False)

    print("Resultados salvos em results/final_autosklearn_results_unb.csv")
