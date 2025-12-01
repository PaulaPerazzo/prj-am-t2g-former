# import os
# import json
# import random
# import numpy as np
# import pandas as pd
# import time

# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, TensorDataset

# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import KFold

# import optuna

# from bin import T2GFormer

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ============================================================
# # Utilities
# # ============================================================

# def seed_everything(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
    
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False


# def load_dataset(dataset_name):
#     train = pd.read_csv(f"train_datasets/{dataset_name}.csv")
#     val   = pd.read_csv(f"val_datasets/{dataset_name}.csv")
#     test  = pd.read_csv(f"test_datasets/{dataset_name}.csv")

#     return train, val, test


# def preprocess(train, val, test):
#     y_train = train["current_target_class"]
#     y_val   = val["current_target_class"]
#     y_test  = test["current_target_class"]

#     le = LabelEncoder()
#     y_train = le.fit_transform(y_train)
#     y_val   = le.transform(y_val)
#     y_test  = le.transform(y_test)

#     cat_cols = train.select_dtypes(include="object").columns.tolist()
#     num_cols = train.select_dtypes(include=np.number).drop(columns=["current_target_class"]).columns.tolist()

#     if len(cat_cols) > 0:
#         cat_encoders = {}

#         for c in cat_cols:
#             enc = LabelEncoder()
#             train[c] = enc.fit_transform(train[c].astype(str))
#             val[c]   = enc.transform(val[c].astype(str))
#             test[c]  = enc.transform(test[c].astype(str))
#             cat_encoders[c] = enc

#     # numéricas → StandardScaler
#     scaler = StandardScaler()
#     train[num_cols] = scaler.fit_transform(train[num_cols])
#     val[num_cols]   = scaler.transform(val[num_cols])
#     test[num_cols]  = scaler.transform(test[num_cols])

#     X_train_num = train[num_cols].values.astype(np.float32)
#     X_val_num   = val[num_cols].values.astype(np.float32)
#     X_test_num  = test[num_cols].values.astype(np.float32)

#     if len(cat_cols) > 0:
#         X_train_cat = train[cat_cols].values.astype(np.int64)
#         X_val_cat   = val[cat_cols].values.astype(np.int64)
#         X_test_cat  = test[cat_cols].values.astype(np.int64)

#     else:
#         X_train_cat = X_val_cat = X_test_cat = None

#     return (
#         X_train_num, X_val_num, X_test_num,
#         X_train_cat, X_val_cat, X_test_cat,
#         y_train, y_val, y_test,
#         num_cols, cat_cols
#     )


# # ============================================================
# # Training Utilities
# # ============================================================

# def make_loaders(X_num, X_cat, y, batch_size):
#     if X_cat is None:
#         ds = TensorDataset(
#             torch.tensor(X_num, dtype=torch.float32, device=DEVICE),
#             torch.tensor(y, dtype=torch.long, device=DEVICE)
#         )

#     else:
#         ds = TensorDataset(
#             torch.tensor(X_num, dtype=torch.float32, device=DEVICE),
#             torch.tensor(X_cat, dtype=torch.long, device=DEVICE),
#             torch.tensor(y, dtype=torch.long, device=DEVICE)
#         )

#     return DataLoader(ds, batch_size=batch_size, shuffle=True)


# def forward_model(model, batch):
#     if len(batch) == 2:
#         x_num, y = batch
#         x_cat = None

#     else:
#         x_num, x_cat, y = batch

#     out = model(x_num, x_cat)

#     return out, y


# # ============================================================
# # Objective (Optuna)
# # ============================================================

# def objective(trial, dataset_name):
#     train_df, val_df_unused, test_df_unused = load_dataset(dataset_name)

#     (
#         X_all_num, _, _,
#         X_all_cat, _, _,
#         y_all, _, _,
#         num_cols, cat_cols
#     ) = preprocess(train_df.copy(), train_df.copy(), train_df.copy())

#     n_num = len(num_cols)
#     n_cat = len(cat_cols)

#     # Configuração do modelo
#     cfg = {
#         "d_numerical": n_num,
#         "categories": [int(train_df[c].nunique()) for c in cat_cols] if n_cat > 0 else None,
#         "d_out": len(np.unique(y_all)),

#         "d_token": trial.suggest_int("d_token", 64, 256, step=32),
#         "n_layers": trial.suggest_int("n_layers", 2, 4),
#         "n_heads": 8,
#         "attention_dropout": trial.suggest_float("attention_dropout", 0.0, 0.5),
#         "ffn_dropout": trial.suggest_float("ffn_dropout", 0.0, 0.5),
#         "d_ffn_factor": trial.suggest_float("d_ffn_factor", 1.0, 3.0),

#         "token_bias": True,
#         "residual_dropout": 0.0,
#         "activation": "reglu",
#         "prenormalization": True,
#         "initialization": "kaiming",
#         "kv_compression": None,
#         "kv_compression_sharing": None,
#     }

#     lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
#     loss_fn = F.cross_entropy

#     kf = KFold(n_splits=10, shuffle=True, random_state=42)
#     fold_scores = []

#     for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(X_all_num)):
#         # Subsets numéricos
#         X_train_num = X_all_num[train_idx]
#         X_valid_num = X_all_num[valid_idx]

#         # Subsets categóricos
#         X_train_cat = X_all_cat[train_idx] if X_all_cat is not None else None
#         X_valid_cat = X_all_cat[valid_idx] if X_all_cat is not None else None

#         y_train = y_all[train_idx]
#         y_valid = y_all[valid_idx]

#         train_loader = make_loaders(X_train_num, X_train_cat, y_train, batch_size=256)
#         valid_loader = make_loaders(X_valid_num, X_valid_cat, y_valid, batch_size=2048)

#         # Modelo deve ser recriado a cada fold
#         model = T2GFormer(**cfg).to(DEVICE)
#         optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

#         # Early stopping
#         best_val = -np.inf
#         patience = 10
#         no_improve = 0

#         # k fold
#         for epoch in range(200):
#             # --- TREINO ---
#             model.train()
            
#             for batch in train_loader:
#                 optimizer.zero_grad()
#                 preds, y = forward_model(model, batch)
#                 loss = loss_fn(preds, y)
#                 loss.backward()
#                 optimizer.step()

#             # --- VALIDAÇÃO ---
#             model.eval()
#             preds_list, y_list = [], []

#             with torch.no_grad():
#                 for batch in valid_loader:
#                     preds, y = forward_model(model, batch)
#                     preds_list.append(torch.argmax(preds, dim=1).cpu().numpy())
#                     y_list.append(y.cpu().numpy())

#             preds_all = np.concatenate(preds_list)
#             y_all_fold = np.concatenate(y_list)

#             acc = (preds_all == y_all_fold).mean()

#             if acc > best_val:
#                 best_val = acc
#                 no_improve = 0
            
#             else:
#                 no_improve += 1
            
#                 if no_improve >= patience:
#                     break

#         # fim do treinamento do fold
#         fold_scores.append(best_val)

#         # report para pruning do Optuna
#         trial.report(best_val, fold_idx)
        
#         if trial.should_prune():
#             raise optuna.TrialPruned()

#     # Score final do trial = média dos 10 folds
#     return float(np.mean(fold_scores))

# # ============================================================
# # MAIN
# # ============================================================

# def run_optuna(dataset_name, n_trials=50):
#     study = optuna.create_study(direction="maximize")
#     study.optimize(lambda trial: objective(trial, dataset_name), n_trials=n_trials)

#     print("\nBest value:", study.best_value)
#     print("Best params:", study.best_params)

#     os.makedirs("optuna_results", exist_ok=True)
    
#     json.dump(
#         {"best_value": study.best_value, "best_params": study.best_params},
#         open(f"optuna_results/{dataset_name}_optuna.json", "w"),
#         indent=4
#     )


# if __name__ == "__main__":
#     import argparse

#     # dataset_list = [
#     #     'credit-approval', 'dresses-sales', 'mfeat-morphological', 'vehicle', 'banknote-authentication', 
#     #     'analcatdata_dmft', 'MiceProtein', 'cylinder-bands', 'semeion', 'cnae-9', 'vowel'
#     # ]

#     # remaining ones
#     dataset_list = [
#         'semeion', 'cnae-9', 'vowel'
#     ]

#     init_time = time.time()

#     for dataset in dataset_list:    
#         parser = argparse.ArgumentParser()
#         parser.add_argument("--dataset", type=str, default=dataset)
#         parser.add_argument("--trials", type=int, default=10)
#         args = parser.parse_args()

#         seed_everything(42)
#         run_optuna(args.dataset, args.trials)
    
#     final_time = time.time()
#     total_time = final_time - init_time

#     print(f"time spent running job: {total_time}")


import os
import json
import random
import numpy as np
import pandas as pd
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold

import optuna

from bin import T2GFormer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_dataset(dataset_name):
    train = pd.read_csv(f"train_datasets/{dataset_name}.csv")
    val   = pd.read_csv(f"val_datasets/{dataset_name}.csv")
    test  = pd.read_csv(f"test_datasets/{dataset_name}.csv")

    return train, val, test


def preprocess(train, val, test):
    y_train = train["current_target_class"]
    y_val   = val["current_target_class"]
    y_test  = test["current_target_class"]

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_val   = le.transform(y_val)
    y_test  = le.transform(y_test)

    cat_cols = train.select_dtypes(include="object").columns.tolist()

    num_cols = (
        train.select_dtypes(include=np.number)
        .drop(columns=["current_target_class"])
        .columns
        .tolist()
    )

    if len(cat_cols) > 0:
        cat_encoders = {}

        for c in cat_cols:
            enc = LabelEncoder()
            train[c] = enc.fit_transform(train[c].astype(str))
            val[c]   = enc.transform(val[c].astype(str))
            test[c]  = enc.transform(test[c].astype(str))
            cat_encoders[c] = enc

    # numéricas → StandardScaler
    scaler = StandardScaler()
    train[num_cols] = scaler.fit_transform(train[num_cols])
    val[num_cols]   = scaler.transform(val[num_cols])
    test[num_cols]  = scaler.transform(test[num_cols])

    X_train_num = train[num_cols].values.astype(np.float32)
    X_val_num   = val[num_cols].values.astype(np.float32)
    X_test_num  = test[num_cols].values.astype(np.float32)

    if len(cat_cols) > 0:
        X_train_cat = train[cat_cols].values.astype(np.int64)
        X_val_cat   = val[cat_cols].values.astype(np.int64)
        X_test_cat  = test[cat_cols].values.astype(np.int64)
    else:
        X_train_cat = X_val_cat = X_test_cat = None

    return (
        X_train_num, X_val_num, X_test_num,
        X_train_cat, X_val_cat, X_test_cat,
        y_train, y_val, y_test,
        num_cols, cat_cols
    )


def prepare_features_for_kfold(train_df):
    y = train_df["current_target_class"].values
    le_y = LabelEncoder()
    y_all = le_y.fit_transform(y)

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
            enc = LabelEncoder()
            df[c] = enc.fit_transform(df[c].astype(str))

        X_all_cat = df[cat_cols].values.astype(np.int64)
    else:
        X_all_cat = None

    X_all_num_raw = df[num_cols].values.astype(np.float32)

    return X_all_num_raw, X_all_cat, y_all, num_cols, cat_cols


def make_loaders(X_num, X_cat, y, batch_size, shuffle=True):
    """
    Cria DataLoader com tensores na CPU.
    O envio para DEVICE (GPU/CPU) é feito no loop de treino.
    """
    x_num_tensor = torch.tensor(X_num, dtype=torch.float32)

    y_tensor = torch.tensor(y, dtype=torch.long)

    if X_cat is None:
        ds = TensorDataset(x_num_tensor, y_tensor)
    else:
        x_cat_tensor = torch.tensor(X_cat, dtype=torch.long)
        ds = TensorDataset(x_num_tensor, x_cat_tensor, y_tensor)

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def move_batch_to_device(batch, device):
    if isinstance(batch, (list, tuple)):
        return tuple(t.to(device) for t in batch)

    return batch.to(device)


def forward_model(model, batch):
    """
    batch já deve estar no DEVICE aqui.
    """
    if len(batch) == 2:
        x_num, y = batch
        x_cat = None
    else:
        x_num, x_cat, y = batch

    out = model(x_num, x_cat)

    return out, y


def objective(trial, dataset_name):
    train_df, _, _ = load_dataset(dataset_name)

    (
        X_all_num_raw,
        X_all_cat,
        y_all,
        num_cols,
        cat_cols,
    ) = prepare_features_for_kfold(train_df.copy())

    n_num = len(num_cols)
    n_cat = len(cat_cols)

    cfg = {
        "d_numerical": n_num,
        "categories": [int(train_df[c].nunique()) for c in cat_cols] if n_cat > 0 else None,
        "d_out": len(np.unique(y_all)),

        "d_token": trial.suggest_int("d_token", 64, 256, step=32),
        "n_layers": trial.suggest_int("n_layers", 2, 4),
        "n_heads": 8,
        "attention_dropout": trial.suggest_float("attention_dropout", 0.0, 0.5),
        "ffn_dropout": trial.suggest_float("ffn_dropout", 0.0, 0.5),
        "d_ffn_factor": trial.suggest_float("d_ffn_factor", 1.0, 3.0),

        "token_bias": True,
        "residual_dropout": 0.0,
        "activation": "reglu",
        "prenormalization": True,
        "initialization": "kaiming",
        "kv_compression": None,
        "kv_compression_sharing": None,
    }

    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    loss_fn = F.cross_entropy

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_scores = []

    for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(X_all_num_raw)):
        # Numéricas cruas para o fold
        X_train_raw = X_all_num_raw[train_idx]
        X_valid_raw = X_all_num_raw[valid_idx]

        # StandardScaler por fold (fit só no treino do fold)
        scaler = StandardScaler()
        X_train_num = scaler.fit_transform(X_train_raw)
        X_valid_num = scaler.transform(X_valid_raw)

        # Categóricas
        X_train_cat = X_all_cat[train_idx] if X_all_cat is not None else None
        X_valid_cat = X_all_cat[valid_idx] if X_all_cat is not None else None

        y_train = y_all[train_idx]
        y_valid = y_all[valid_idx]

        train_loader = make_loaders(X_train_num, X_train_cat, y_train, batch_size=256, shuffle=True)
        valid_loader = make_loaders(X_valid_num, X_valid_cat, y_valid, batch_size=2048, shuffle=False)

        # Modelo deve ser recriado a cada fold
        model = T2GFormer(**cfg).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        # Early stopping com base em ACC (como você pediu)
        best_val = -np.inf
        patience = 3
        no_improve = 0

        for epoch in range(200):
            model.train()

            for batch in train_loader:
                batch = move_batch_to_device(batch, DEVICE)
                optimizer.zero_grad()
                preds, y = forward_model(model, batch)
                loss = loss_fn(preds, y)
                loss.backward()
                optimizer.step()

            model.eval()
            preds_list, y_list = [], []

            with torch.no_grad():
                for batch in valid_loader:
                    batch = move_batch_to_device(batch, DEVICE)
                    preds, y = forward_model(model, batch)
                    preds_list.append(torch.argmax(preds, dim=1).cpu().numpy())
                    y_list.append(y.cpu().numpy())

            preds_all = np.concatenate(preds_list)
            y_all_fold = np.concatenate(y_list)

            acc = (preds_all == y_all_fold).mean()

            if acc > best_val:
                best_val = acc
                no_improve = 0
            else:
                no_improve += 1

                if no_improve >= patience:
                    break

        # fim do treinamento do fold
        fold_scores.append(best_val)

        # report para pruning do Optuna
        trial.report(best_val, fold_idx)

        if trial.should_prune():
            raise optuna.TrialPruned()

    # Score final do trial = média dos 10 folds
    return float(np.mean(fold_scores))


# ============================================================
# MAIN
# ============================================================

def run_optuna(dataset_name, n_trials=50):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, dataset_name), n_trials=n_trials)

    print(f"\n[{dataset_name}] Best value:", study.best_value)
    print(f"[{dataset_name}] Best params:", study.best_params)

    os.makedirs("optuna_results", exist_ok=True)
    with open(f"optuna_results/{dataset_name}_optuna.json", "w") as f:
        json.dump(
            {"best_value": study.best_value, "best_params": study.best_params},
            f,
            indent=4
        )


if __name__ == "__main__":
    # Exemplo: rodando apenas estes (como você tinha no final)
    dataset_list = [
        "semeion",
        "cnae-9",
        "vowel",
    ]

    init_time = time.time()
    seed_everything(42)

    for dataset in dataset_list:
        print(f"\n========== Rodando Optuna para: {dataset} ==========")
        run_optuna(dataset, n_trials=10)

    total_time = time.time() - init_time
    print(f"\nTempo total de execução: {total_time:.2f} segundos")
