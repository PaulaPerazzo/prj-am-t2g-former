import os
import json
import time
import numpy as np
import pandas as pd
import optuna
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from imblearn.metrics import geometric_mean_score

SEED = 42

# ==========================================================
# 1. Split dataset
# ==========================================================
def split_dataset(file_path, test_size=0.3):
  df = pd.read_csv(file_path)
  train_df, test_df = train_test_split(df, test_size=test_size, random_state=SEED, stratify=df.iloc[:, -1])
  return train_df, test_df


# ==========================================================
# 2. Métricas por fold (validação)
# ==========================================================
def fold_metrics(model, X_val, y_val):
  y_pred = model.predict(X_val)
  y_proba = model.predict_proba(X_val)

  metrics = {
    "accuracy": accuracy_score(y_val, y_pred),
    "gmean": geometric_mean_score(y_val, y_pred),
    "cross_entropy": log_loss(y_val, y_proba)
  }

  # AUC-OVO
  classes = np.unique(y_val)
  if len(classes) == 2:
    y_score = y_proba[:, 1]
    auc = roc_auc_score(y_val, y_score)
  else:
    auc = roc_auc_score(y_val, y_proba, multi_class="ovo", average="macro")

  metrics["auc_ovo"] = auc
  return metrics


# ==========================================================
# 3. Cross-validation completa (retorna as MÉDIAS DAS MÉTRICAS)
# ==========================================================
def cross_val_full_metrics(model, X, y, n_splits=10):
  cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

  all_metrics = {
    "accuracy": [],
    "gmean": [],
    "cross_entropy": [],
    "auc_ovo": []
  }

  for train_idx, val_idx in cv.split(X, y):
    X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
    y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

    model.fit(X_train_cv, y_train_cv)
    m = fold_metrics(model, X_val_cv, y_val_cv)

    for key in all_metrics:
      all_metrics[key].append(m[key])

  return {key: float(np.mean(values)) for key, values in all_metrics.items()}


# ==========================================================
# 4. Optuna Hyperparameter Search
# ==========================================================
def run_optuna_search(X_train, y_train, metric, n_trials=50):
  def objective(trial):
    params = {
      "num_leaves": trial.suggest_categorical("num_leaves", [10, 31, 63, 127, 255]),
      "learning_rate": trial.suggest_categorical("learning_rate", [0.1, 0.03, 0.01, 0.003, 0.001]),
      "n_estimators": trial.suggest_categorical("n_estimators", [100, 300, 500, 800, 1200]),
      "max_depth": trial.suggest_categorical("max_depth", [3, 5, 7, 10, -1]),
      "min_child_samples": trial.suggest_categorical("min_child_samples", [5, 10, 20, 30]),
      "random_state": SEED
    }

    model = LGBMClassifier(**params, verbose=-1)

    cv_results = cross_val_full_metrics(model, X_train, y_train, n_splits=10)

    # métrica a otimizar
    return cv_results["accuracy"] if metric == "accuracy" else cv_results["gmean"]

  study = optuna.create_study(direction="maximize")
  study.optimize(objective, n_trials=n_trials)

  return study.best_params


# ==========================================================
# 5. Métricas no TREINO/TESTE
# ==========================================================
def compute_metrics(model, X, y):
  y_pred = model.predict(X)
  y_proba = model.predict_proba(X)

  metrics = {
    "accuracy": accuracy_score(y, y_pred),
    "gmean": geometric_mean_score(y, y_pred),
    "cross_entropy": log_loss(y, y_proba)
  }

  # AUC one-vs-one
  classes = np.unique(y)
  if len(classes) == 2:
    y_score = y_proba[:, 1]
    auc = roc_auc_score(y, y_score)
  else:
    auc = roc_auc_score(y, y_proba, multi_class="ovo", average="macro")

  metrics["auc_ovo"] = auc
  return metrics


# ==========================================================
# 6. Métrica por dataset
# ==========================================================
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


# ==========================================================
# 7. LOOP PRINCIPAL
# ==========================================================
results = {}

for filename in os.listdir('datasets'):
  print(f'Analyzing {filename}')
  dataset_name = filename.replace('.csv', '')

  train_set, test_set = split_dataset(os.path.join('datasets', filename))

  target = 'current_target_class'
  metric = metric_report[dataset_name]

  X_train, y_train = train_set.drop(columns=[target]), train_set[target]
  X_test,  y_test  = test_set.drop(columns=[target]),  test_set[target]

  # -----------------------------------------------
  # OPTUNA TUNING
  # -----------------------------------------------
  start_tune = time.time()
  best_params = run_optuna_search(X_train, y_train, metric, n_trials=50)
  tune_time = time.time() - start_tune

  # -----------------------------------------------
  # MÉTRICAS DE TREINO = MÉDIA DA CV COM BEST PARAMS
  # -----------------------------------------------
  cv_model = LGBMClassifier(**best_params, verbose=-1)
  train_metrics = cross_val_full_metrics(cv_model, X_train, y_train, n_splits=10)

  # -----------------------------------------------
  # MODEL FINAL
  # -----------------------------------------------
  model = LGBMClassifier(**best_params, verbose=-1)

  start_train = time.time()
  model.fit(X_train, y_train)
  train_time = time.time() - start_train

  start_pred = time.time()
  model.predict(X_test)
  predict_time = time.time() - start_pred

  # -----------------------------------------------
  # METRICS
  # -----------------------------------------------
  test_metrics  = compute_metrics(model, X_test, y_test)

  results[dataset_name] = {
    "best_params": best_params,
    "train_metrics": train_metrics,
    "test_metrics": test_metrics,
    "total_time_seconds": tune_time + train_time + predict_time
  }

# ==========================================================
# 8. Salvar resultados
# ==========================================================
with open('light_gbm_optuna_results.json', 'w') as f:
  json.dump(results, f, indent=2)

print("\nDONE! Results saved to light_gbm_optuna_results.json\n")
