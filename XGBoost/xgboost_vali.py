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

    if ds_name in desbalanceados:
        scoring = gmean_scorer
        best_params_preset = "gbm_xgb"   # ou conforme seu preset/categorização
    else:
        scoring = 'accuracy'
        best_params_preset = "gbm_xgb"

    df_train = pd.read_csv(os.path.join(train_path, dsfile))
    df_test  = pd.read_csv(os.path.join(test_path, dsfile))
    
    X_train = df_train.iloc[:, :-1]
    y_train = df_train.iloc[:, -1]
    X_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1]
    
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Tempo de tuning (GridSearchCV)
    start_tune = time.time()
    grid = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=0
    )
    grid.fit(X_train, y_train)
    tuning_time = time.time() - start_tune
    best_xgb = grid.best_estimator_

    # Tempo de treinamento do melhor modelo (refit final)
    start_train = time.time()
    best_xgb.fit(X_train, y_train)
    training_time = time.time() - start_train

    # Tempo de predição teste
    start_pred = time.time()
    y_test_pred = best_xgb.predict(X_test)
    y_test_pred_proba = best_xgb.predict_proba(X_test)
    predict_time = time.time() - start_pred

    # Tempo total = soma dos tempos
    total_time = tuning_time + training_time + predict_time

    # Métricas treino
    y_train_pred = best_xgb.predict(X_train)
    y_train_pred_proba = best_xgb.predict_proba(X_train)
    train_metrics_accuracy = accuracy_score(y_train, y_train_pred)
    train_metrics_gmean = gmean_score(y_train, y_train_pred)
    train_metrics_cross_entropy = log_loss(y_train, y_train_pred_proba)
    try:
        train_metrics_auc_ovo = roc_auc_score(y_train, y_train_pred_proba, multi_class='ovo')
    except:
        train_metrics_auc_ovo = np.nan

    # Métricas teste
    test_metrics_accuracy = accuracy_score(y_test, y_test_pred)
    test_metrics_gmean    = gmean_score(y_test, y_test_pred)
    test_metrics_cross_entropy = log_loss(y_test, y_test_pred_proba)
    try:
        test_metrics_auc_ovo = roc_auc_score(y_test, y_test_pred_proba, multi_class='ovo')
    except:
        test_metrics_auc_ovo = np.nan

    results.append({
        "dataset": dsfile,
        "total_time": total_time,
        "tuning_time": tuning_time,
        "training_time": training_time,
        "predict_time": predict_time,
        "best_params_preset": best_params_preset,
        "best_params_model_set": grid.best_params_,
        "train_metrics_accuracy": train_metrics_accuracy,
        "train_metrics_gmean": train_metrics_gmean,
        "train_metrics_cross_entropy": train_metrics_cross_entropy,
        "train_metrics_auc_ovo": train_metrics_auc_ovo,
        "test_metrics_accuracy": test_metrics_accuracy,
        "test_metrics_gmean": test_metrics_gmean,
        "test_metrics_cross_entropy": test_metrics_cross_entropy,
        "test_metrics_auc_ovo": test_metrics_auc_ovo
    })

df_final = pd.DataFrame(results, columns=[
    "dataset","total_time","tuning_time","training_time","predict_time",
    "best_params_preset","best_params_model_set",
    "train_metrics_accuracy","train_metrics_gmean","train_metrics_cross_entropy","train_metrics_auc_ovo",
    "test_metrics_accuracy","test_metrics_gmean","test_metrics_cross_entropy","test_metrics_auc_ovo"
])
df_final.to_csv("xgboost_todos_resultados.csv", index=False)
print("Resultados salvos em xgboost_todos_resultados.csv")