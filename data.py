import pandas as pd

# 출력 제한 해제
pd.set_option('display.max_colwidth', None)

# 데이터 로드
df_train = pd.read_parquet(r"C:\Users\dsng3\Documents\GitHub\DoF_Experiment\data\train-00000-of-00001.parquet")
df_valid = pd.read_parquet(r"C:\Users\dsng3\Documents\GitHub\DoF_Experiment\data\validation-00000-of-00001.parquet")
df_test = pd.read_parquet(r"C:\Users\dsng3\Documents\GitHub\DoF_Experiment\data\test-00000-of-00001.parquet")

def show_sentence_by_index(idx):
    """Train / Validation / Test 각 데이터셋에서 idx에 해당하는 sentence를 출력"""
    if idx in df_train.index:
        print(f"[Train] idx={idx}:")
        print(df_train.loc[idx, "sentence"], "\n")
    if idx in df_valid.index:
        print(f"[Validation] idx={idx}:")
        print(df_valid.loc[idx, "sentence"], "\n")
    if idx in df_test.index:
        print(f"[Test] idx={idx}:")
        print(df_test.loc[idx, "sentence"], "\n")

# 예시: 144번 인덱스 문장 보기
show_sentence_by_index(144)
