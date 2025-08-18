import pandas as pd

# 데이터 로드
df_train = pd.read_parquet(r"C:\Users\dsng3\Documents\GitHub\DoF_Experiment\data\train-00000-of-00001.parquet")
df_valid = pd.read_parquet(r"C:\Users\dsng3\Documents\GitHub\DoF_Experiment\data\validation-00000-of-00001.parquet")
df_test = pd.read_parquet(r"C:\Users\dsng3\Documents\GitHub\DoF_Experiment\data\test-00000-of-00001.parquet")

def show_sentence_by_index(idx):
    if idx in df_train.index:
        print(f"[Train] idx={idx}:")
        print(df_train.loc[idx, "sentence"], "\n")
    if idx in df_valid.index:
        print(f"[Validation] idx={idx}:")
        print(df_valid.loc[idx, "sentence"], "\n")
    if idx in df_test.index:
        print(f"[Test] idx={idx}:")
        print(df_test.loc[idx, "sentence"], "\n")

show_sentence_by_index(144)
