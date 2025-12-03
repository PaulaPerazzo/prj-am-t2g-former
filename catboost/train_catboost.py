import os
import json
import time
import warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

SEED = 42


def ensure_dirs():
    os.makedirs("trained_models_catboost", exist_ok=True)
    os.makedirs("results", exist_ok=True)


def load_best_params(dataset_name: str):
    """Carrega hiperparâmetros do Optuna."""
    path = f"optuna_results/{dataset_name}_catboost_best.json"
    with open(path, "r") as f:
        data = json.load(f)
    return data["best_params"], data.get("tune_time", np.nan)


def load_dataset(dataset_name: str):
    """Carrega os datasets já separados."""
    train = pd.read_csv(f"train_datasets/{dataset_name}.csv")
    test = pd.read_csv(f"test_datasets/{dataset_name}.csv")
    return train, test


def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Preprocessamento padronizado para treino e teste."""

    # Fit LabelEncoder no treino
    le_target = LabelEncoder()
    y_train = le_target.fit_transform(train_df["current_target_class"].values)
    y_test = le_target.transform(test_df["current_target_class"].values)

    cat_cols = train_df.select_dtypes(include="object").columns.tolist()
    num_cols = (
        train_df.select_dtypes(include=np.number)
        .drop(columns=["current_target_class"])
        .columns.tolist()
    )

    train_copy = train_df.copy()
    test_copy = test_df.copy()

    # Codificar categóricas
    cat_indices = []
    cat_encoders = {}

    if len(cat_cols) > 0:
        for c in cat_cols:
            enc = LabelEncoder()
            train_copy[c] = enc.fit_transform(train_copy[c].astype(str))

            # Handle unseen categories in test
            test_vals = test_copy[c].astype(str)
            test_copy[c] = test_vals.apply(
                lambda x: enc.transform([x])[0] if x in enc.classes_ else -1
            )
            cat_encoders[c] = enc

        cat_indices = list(range(len(num_cols), len(num_cols) + len(cat_cols)))

    # Features numéricas
    X_train_num = train_copy[num_cols].values.astype(np.float32)
    X_test_num = test_copy[num_cols].values.astype(np.float32)

    # Normalização
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train_num)
    X_test_num = scaler.transform(X_test_num)

    # Concatenar
    if len(cat_cols) > 0:
        X_train_cat = train_copy[cat_cols].values.astype(np.int32)
        X_test_cat = test_copy[cat_cols].values.astype(np.int32)
        X_train = np.hstack([X_train_num, X_train_cat])
        X_test = np.hstack([X_test_num, X_test_cat])
    else:
        X_train = X_train_num
        X_test = X_test_num

    return X_train, X_test, y_train, y_test, cat_indices, le_target


def gmean_score(y_true, y_pred, eps=1e-9):
    """G-Mean: média geométrica dos recalls por classe."""
    classes = np.unique(y_true)
    recalls = []

    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        recalls.append(tp / (tp + fn + eps))

    return float(np.prod(recalls) ** (1.0 / len(recalls)))


def compute_auc_ovo(y_true, probs):
    """Cálculo robusto do AUC OVO."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    unique_classes = np.unique(y_true)

    if len(unique_classes) < 2:
        return np.nan

    try:
        if len(unique_classes) > 2:
            return roc_auc_score(
                y_true,
                probs,
                multi_class="ovo",
                average="macro",
                labels=unique_classes
            )
        else:
            return roc_auc_score(y_true, probs[:, 1])
    except Exception:
        return np.nan


def evaluate_split(model, X, y):
    """Avaliação completa em um split (train/test)."""
    start = time.time()
    probs = model.predict_proba(X)
    tempo_predict = time.time() - start

    probs = np.asarray(probs)
    y = np.asarray(y)

    preds = np.argmax(probs, axis=1)

    return {
        "auc_ovo": compute_auc_ovo(y, probs),
        "mean_acc": accuracy_score(y, preds),
        "g_mean": gmean_score(y, preds),
        "mean_cross_entropy": log_loss(y, probs),
        "tempo_predict": tempo_predict,
    }


def train_final_model(dataset_name: str):
    """Treina modelo final com melhores hiperparâmetros."""
    print(f"\n{'='*60}")
    print(f"Treinando CatBoost para: {dataset_name}")
    print(f"{'='*60}")

    # Carregar melhores parâmetros
    best_params, tempo_tune = load_best_params(dataset_name)
    print(f"Melhores hiperparâmetros: {best_params}")

    # Carregar dados
    train_df, test_df = load_dataset(dataset_name)

    # Preprocessar
    X_train, X_test, y_train, y_test, cat_indices, le_target = preprocess(
        train_df, test_df
    )

    print(f"  Train: {X_train.shape[0]} amostras")
    print(f"  Test: {X_test.shape[0]} amostras")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Classes: {len(np.unique(y_train))}")

    # Configurar modelo
    model_params = {
        **best_params,
        "random_seed": SEED,
        "verbose": False,
        "thread_count": -1,  # Usar todos os cores para treino final
        "task_type": "CPU",
    }

    model = CatBoostClassifier(**model_params)

    # Treinar
    start_train = time.time()

    if len(cat_indices) > 0:
        model.fit(X_train, y_train, cat_features=cat_indices, verbose=False)
    else:
        model.fit(X_train, y_train, verbose=False)

    tempo_train = time.time() - start_train
    print(f"  Tempo de treino: {tempo_train:.2f}s")

    # Salvar modelo
    model_path = f"trained_models_catboost/{dataset_name}_catboost.cbm"
    model.save_model(model_path)
    print(f"  Modelo salvo: {model_path}")

    # Avaliar
    train_metrics = evaluate_split(model, X_train, y_train)
    test_metrics = evaluate_split(model, X_test, y_test)

    print(f"\n  Resultados no TREINO:")
    print(f"    AUC OVO: {train_metrics['auc_ovo']:.4f}")
    print(f"    Accuracy: {train_metrics['mean_acc']:.4f}")
    print(f"    G-Mean: {train_metrics['g_mean']:.4f}")
    print(f"    Cross-Entropy: {train_metrics['mean_cross_entropy']:.4f}")

    print(f"\n  Resultados no TESTE:")
    print(f"    AUC OVO: {test_metrics['auc_ovo']:.4f}")
    print(f"    Accuracy: {test_metrics['mean_acc']:.4f}")
    print(f"    G-Mean: {test_metrics['g_mean']:.4f}")
    print(f"    Cross-Entropy: {test_metrics['mean_cross_entropy']:.4f}")

    # Construir saída
    rows = []

    for split, metrics in [("train", train_metrics), ("test", test_metrics)]:
        rows.append({
            "split": split,
            "nome_modelo": "CatBoost",
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


if __name__ == "__main__":
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

    all_rows = []

    for ds in dataset_list:
        try:
            rows = train_final_model(ds)
            all_rows.extend(rows)
        except FileNotFoundError:
            print(f"\n[AVISO] Parâmetros não encontrados para {ds}. Execute tune_catboost.py primeiro.")
            continue
        except Exception as e:
            print(f"\n[ERRO] Dataset {ds}: {e}")
            continue

    # Salvar resultados
    if len(all_rows) > 0:
        df = pd.DataFrame(all_rows)
        output_path = "results/final_catboost_results.csv"
        df.to_csv(output_path, index=False)
        print(f"\n{'='*60}")
        print(f"Resultados salvos em: {output_path}")
        print(f"{'='*60}")

        # Mostrar resumo
        print("\nResumo dos resultados no TESTE:")
        test_df = df[df["split"] == "test"][["dataset", "auc_ovo", "mean_acc", "g_mean"]]
        print(test_df.to_string(index=False))
