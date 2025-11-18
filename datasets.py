import os
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_DIR = "datasets"
TRAIN_DIR = "train_datasets"
TEST_DIR = "test_datasets"
SEED = 42

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

def split_dataset(file_path, filename):
    print(f"processing {filename}...")

    df = pd.read_csv(file_path)

    train_df, test_df = train_test_split(df, test_size=0.3, random_state=SEED)

    train_df.to_csv(os.path.join(TRAIN_DIR, filename), index=False)
    test_df.to_csv(os.path.join(TEST_DIR, filename), index=False)

def main():
    count_dataset = 0

    for filename in os.listdir(INPUT_DIR):
        count_dataset += 1

        file_path = os.path.join(INPUT_DIR, filename)

        if os.path.isfile(file_path) and filename.endswith(".csv"):
            split_dataset(file_path, filename)

    print("train test split completed.")
    print(f"{count_dataset} datasets processed.")

if __name__ == "__main__":
    main()
