import pandas as pd

FILE = "LightGBM/results.csv"
NEW_FILE = "LightGBM/converted_results.csv"
MODEL = "LightGBM"

df = pd.read_csv(FILE)

linhas = []

for _, row in df.iterrows():
    dataset = row["dataset"]
    tempo_tune = row["tuning_time"]
    tempo_train = row["training_time"]
    tempo_predict = row["predict_time"]

    linhas.append({
        "split": "train",
        "nome_modelo": MODEL,
        "dataset": dataset,
        "tempo_tune": tempo_tune,
        "tempo_train": tempo_train,
        "auc_ovo": row["train_metrics_auc_ovo"],
        "mean_acc": row["train_metrics_accuracy"],
        "g_mean": row["train_metrics_gmean"],
        "mean_cross_entropy": row["train_metrics_cross_entropy"],
        "tempo_predict": tempo_predict,
    })

    linhas.append({
        "split": "test",
        "nome_modelo": MODEL,
        "dataset": dataset,
        "tempo_tune": tempo_tune,
        "tempo_train": tempo_train,
        "auc_ovo": row["test_metrics_auc_ovo"],
        "mean_acc": row["test_metrics_accuracy"],
        "g_mean": row["test_metrics_gmean"],
        "mean_cross_entropy": row["test_metrics_cross_entropy"],
        "tempo_predict": tempo_predict,
    })

df_novo = pd.DataFrame(linhas)

df_novo.to_csv(NEW_FILE, index=False)

print("Conversão concluída!")
print(f"Arquivo salvo: {NEW_FILE}")
print("Linhas geradas:", len(df_novo))
