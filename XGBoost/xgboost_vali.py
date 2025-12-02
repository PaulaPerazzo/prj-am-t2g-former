import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, confusion_matrix, make_scorer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

desbalanceados = [
    'breast-w','eucalyptus', 'wdbc', 'pc4', 'credit-g', 'cmc', 'blood-transfusion-service-center',
    'pc3', 'car', 'kc2', 'steel-plates-fault', 'balance-scale', 'pc1',
    'tic-tac-toe', 'analcatdata_authorship', 'climate-model-simulation-crashes', 'qsar-biodeg',
    'diabetes', 'ilpd'
]
balanceados = [
    'credit-approval', 'dresses-sales', 'mfeat-morphological', 'vehicle', 'banknote-authentication', 
    'analcatdata_dmft', 'MiceProtein', 'cylinder-bands', 'semeion', 'cnae-9', 'vowel'
]

def gmean_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sensitivities = cm.diagonal() / cm.sum(axis=1)
    gmean = np.prod(sensitivities) ** (1.0 / len(sensitivities))
    return gmean

gmean_scorer = make_scorer(gmean_score, needs_proba=False)

train_path = "train_datasets"
test_path = "test_datasets"

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1]
}

results = []

for dsfile in sorted(os.listdir(train_path)):
    if not dsfile.endswith(".csv"):
        continue

    ds_name = dsfile.replace(".csv", "")
    print(f"Processando: {dsfile}")

    # MÉTRICA A SER OTIMIZADA
    if ds_name in desbalanceados:
        scoring = gmean_scorer
        otimizado_com = "gmean"
    else:
        scoring = 'accuracy'
        otimizado_com = "accuracy"

    # Ler conjunto de treino e teste
    df_train = pd.read_csv(os.path.join(train_path, dsfile))
    df_test  = pd.read_csv(os.path.join(test_path, dsfile))
    
    # Supondo que a última coluna é o target
    X_train = df_train.iloc[:, :-1]
    y_train = df_train.iloc[:, -1]
    X_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1]
    
    # --- Adaptação: garantir que os rótulos começam em zero e são numéricos ---
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    # Instanciar XGBClassifier
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    # Validação cruzada no conjunto de treino
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    # Cronometrar tuning + fit
    start_time = time.time()
    grid.fit(X_train, y_train)
    tempo_total = time.time() - start_time

    best_xgb = grid.best_estimator_

    # Métricas em treino
    y_train_pred = best_xgb.predict(X_train)
    y_train_pred_proba = best_xgb.predict_proba(X_train)
    acc_train = accuracy_score(y_train, y_train_pred)
    try:
        auc_train = roc_auc_score(y_train, y_train_pred_proba, multi_class='ovo')
    except:
        auc_train = np.nan
    gmean_train = gmean_score(y_train, y_train_pred)
    ce_train = log_loss(y_train, y_train_pred_proba)

    # Métricas em teste
    y_test_pred = best_xgb.predict(X_test)
    y_test_pred_proba = best_xgb.predict_proba(X_test)
    acc = accuracy_score(y_test, y_test_pred)
    try:
        auc = roc_auc_score(y_test, y_test_pred_proba, multi_class='ovo')
    except:
        auc = np.nan
    gmean = gmean_score(y_test, y_test_pred)
    ce = log_loss(y_test, y_test_pred_proba)

    results.append({
        "dataset": dsfile,
        "mean_accuracy_test": acc,
        "mean_auc_ovo_test": auc,
        "gmean_test": gmean,
        "cross_entropy_test": ce,
        "mean_accuracy_train": acc_train,
        "mean_auc_ovo_train": auc_train,
        "gmean_train": gmean_train,
        "cross_entropy_train": ce_train,
        "train_time_sec": tempo_total,
        "best_params": grid.best_params_,
        "otimizado_com": otimizado_com
    })

df_final = pd.DataFrame(results)
df_final.to_csv("xgboost_todos_resultados.csv", index=False)
print("Resultados salvos em xgboost_todos_resultados.csv")