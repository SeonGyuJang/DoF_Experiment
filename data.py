import pandas as pd

df_train = pd.read_parquet(r"C:\Users\dsng3\Documents\GitHub\DoF_Experiment\data\train-00000-of-00001.parquet")
#df_valid = pd.read_parquet(r"C:\Users\dsng3\Documents\GitHub\DoF_Experiment\data\validation-00000-of-00001.parquet")
df_test = pd.read_parquet(r"C:\Users\dsng3\Documents\GitHub\DoF_Experiment\data\test-00000-of-00001.parquet")

def show_sentence_by_index(idx):
    if idx in df_train.index:
        print(f"[Train] idx={idx}:")
        print(df_train.loc[idx, "text"], "\n")
    if idx in df_test.index:
        print(f"[Test] idx={idx}:")
        print(df_test.loc[idx, "text"], "\n")

show_sentence_by_index(34)
