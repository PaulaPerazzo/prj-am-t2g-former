import os
import json
import time
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

from tune_t2g_for_unbalanced_data import (
    DEVICE,
    seed_everything,
    load_dataset,
    preprocess,
    make_loaders,
    forward_model,
    gmean_score,
)

from bin import T2GFormer

RESULTS_CSV = "t2gformer_best_configs_results.csv"
MODELS_DIR = "trained_models_t2gformer"
OPTUNA_DIR = "optuna_results"

os.makedirs(MODELS_DIR, exist_ok=True)


def load_best_params(dataset_name, optuna_dir=OPTUNA_DIR):
    path = os.path.join(optuna_dir, f"{dataset_name}_optuna.json")

    with open(path, "r") as f:
        data = json.load(f)
    
    return data["best_params"], data.get("best_value", None)


def compute_auc_ovo(y_true, y_proba):
    """
    AUC OVO (One-vs-One) macro.
    - Para binário, usa AUC clássico com proba da classe positiva.
    - Para multi-classe, usa multi_class='ovo', average='macro'.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    classes = np.unique(y_true)

    if len(classes) == 1:
        # cant calculate AUC
        return np.nan

    if len(classes) == 2:
        return roc_auc_score(y_true, y_proba[:, 1])
    else:
        return roc_auc_score(
            y_true,
            y_proba,
            multi_class="ovo",
            average="macro",
        )


def evaluate_split(model, X_num, X_cat, y_true, batch_size=4096):
    model.eval()

    if X_cat is None:
        ds_tensors = (
            torch.tensor(X_num, dtype=torch.float32, device=DEVICE),
            torch.tensor(y_true, dtype=torch.long, device=DEVICE),
        )
    else:
        ds_tensors = (
            torch.tensor(X_num, dtype=torch.float32, device=DEVICE),
            torch.tensor(X_cat, dtype=torch.long, device=DEVICE),
            torch.tensor(y_true, dtype=torch.long, device=DEVICE),
        )

    ds = torch.utils.data.TensorDataset(*ds_tensors)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_logits = []
    all_targets = []

    start_pred = time.time()

    with torch.no_grad():
        for batch in loader:
            preds, y = forward_model(model, batch)
            all_logits.append(preds.cpu())
            all_targets.append(y.cpu())

    tempo_predict = time.time() - start_pred

    logits = torch.cat(all_logits, dim=0)
    y_true_np = torch.cat(all_targets, dim=0).numpy()

    probs = torch.softmax(logits, dim=1).numpy()
    y_pred = np.argmax(probs, axis=1)

    mean_acc = accuracy_score(y_true_np, y_pred)
    g_mean = gmean_score(y_true_np, y_pred)
    mean_ce = log_loss(y_true_np, probs)
    auc_ovo = compute_auc_ovo(y_true_np, probs)

    return {
        "auc_ovo": float(auc_ovo),
        "mean_acc": float(mean_acc),
        "g_mean": float(g_mean),
        "mean_cross_entropy": float(mean_ce),
        "tempo_predict": float(tempo_predict),
    }


def train_final_model_for_dataset(dataset_name):
    """
    1) Carrega melhor config do Optuna.
    2) Carrega train/val/test originais.
    3) Treina com early stopping olhando G-Mean em val.
    4) Avalia métricas em train e test.
    5) Salva modelo e retorna dicts com resultados e tempos.
    """
    print(f"\n===== Dataset: {dataset_name} =====")

    # optuna hyperparams
    best_params, best_gmean_cv = load_best_params(dataset_name)

    print("Best G-MEAN (CV Optuna):", best_gmean_cv)
    print("Best params:", best_params)

    # 2. datasets
    train_df, val_df, test_df = load_dataset(dataset_name)

    (
        X_train_num, X_val_num, X_test_num,
        X_train_cat, X_val_cat, X_test_cat,
        y_train, y_val, y_test,
        num_cols, cat_cols
    ) = preprocess(train_df.copy(), val_df.copy(), test_df.copy())

    n_num = len(num_cols)
    n_cat = len(cat_cols)
    n_classes = len(np.unique(y_train))

    # 3. configs
    cfg = {
        "d_numerical": n_num,
        "categories": [int(train_df[c].nunique()) for c in cat_cols] if n_cat > 0 else None,
        "d_out": n_classes,
        "d_token": best_params["d_token"],
        "n_layers": best_params["n_layers"],
        "n_heads": 8,
        "attention_dropout": best_params["attention_dropout"],
        "ffn_dropout": best_params["ffn_dropout"],
        "d_ffn_factor": best_params["d_ffn_factor"],
        "token_bias": True,
        "residual_dropout": 0.0,
        "activation": "reglu",
        "prenormalization": True,
        "initialization": "kaiming",
        "kv_compression": None,
        "kv_compression_sharing": None,
    }

    lr = best_params["lr"]
    loss_fn = F.cross_entropy

    # 4. DataLoaders 
    train_loader = make_loaders(X_train_num, X_train_cat, y_train, batch_size=256)
    val_loader = make_loaders(X_val_num, X_val_cat, y_val, batch_size=2048)

    # 5. Loop de treino com early stopping em G-Mean (val)
    model = T2GFormer(**cfg).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_gmean = -np.inf
    best_state_dict = None
    patience = 10
    no_improve = 0

    start_train = time.time()

    for epoch in range(200):
        model.train()

        for batch in train_loader:
            optimizer.zero_grad()
            preds, y = forward_model(model, batch)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()

        model.eval()
        preds_list, y_list = [], []

        with torch.no_grad():
            for batch in val_loader:
                preds, y = forward_model(model, batch)
                preds_list.append(torch.argmax(preds, dim=1).cpu().numpy())
                y_list.append(y.cpu().numpy())

        preds_all = np.concatenate(preds_list)
        y_all_val = np.concatenate(y_list)
        gmean_val = gmean_score(y_all_val, preds_all)

        if gmean_val > best_val_gmean:
            best_val_gmean = gmean_val
            best_state_dict = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1

            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}, best G-Mean (val) = {best_val_gmean:.4f}")
                break

    tempo_train = time.time() - start_train

    # Carregar melhor estado (early stopping)
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # 6. evaluate train and test
    train_metrics = evaluate_split(model, X_train_num, X_train_cat, y_train)
    test_metrics = evaluate_split(model, X_test_num, X_test_cat, y_test)

    # 7. save model
    model_path = os.path.join(MODELS_DIR, f"{dataset_name}_t2gformer.pt")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "best_val_gmean": best_val_gmean,
            "dataset_name": dataset_name,
        },
        model_path,
    )

    print(f"Modelo salvo em: {model_path}")

    # -----------------------------
    # 8. Montar dicts finais (train/test)
    # -----------------------------
    # Como não temos tempo de tuning por dataset, deixo NaN (ou None).
    tempo_tune = np.nan

    common_info = {
        "nome_modelo": "t2g-former",
        "dataset": dataset_name,
        "tempo_tune": float(tempo_tune) if not np.isnan(tempo_tune) else np.nan,
        "tempo_train": float(tempo_train),
    }

    row_train = {
        "split": "train",
        **common_info,
        "auc_ovo": train_metrics["auc_ovo"],
        "mean_acc": train_metrics["mean_acc"],
        "g_mean": train_metrics["g_mean"],
        "mean_cross_entropy": train_metrics["mean_cross_entropy"],
        "tempo_predict": train_metrics["tempo_predict"],
    }

    row_test = {
        "split": "test",
        **common_info,
        "auc_ovo": test_metrics["auc_ovo"],
        "mean_acc": test_metrics["mean_acc"],
        "g_mean": test_metrics["g_mean"],
        "mean_cross_entropy": test_metrics["mean_cross_entropy"],
        "tempo_predict": test_metrics["tempo_predict"],
    }

    return row_train, row_test


def append_results_to_csv(rows, csv_path=RESULTS_CSV):
    """
    Salva/append em CSV. Se o arquivo não existe, escreve header.
    """
    df_new = pd.DataFrame(rows)
    
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    
    df_all.to_csv(csv_path, index=False)
    
    print(f"Resultados salvos/atualizados em: {csv_path}")


if __name__ == "__main__":
    seed_everything(42)

    dataset_list = [
        'breast-w','eucalyptus', 'wdbc', 'pc4', 'credit-g', 'cmc', 'blood-transfusion-service-center',
        'pc3', 'car', 'kc2', 'steel-plates-fault', 'balance-scale', 'pc1',
        'tic-tac-toe', 'analcatdata_authorship', 'climate-model-simulation-crashes', 'qsar-biodeg',
        'diabetes', 'ilpd'
    ]

    all_rows = []

    for ds in dataset_list:
        row_train, row_test = train_final_model_for_dataset(ds)
        all_rows.extend([row_train, row_test])

    append_results_to_csv(all_rows, RESULTS_CSV)
