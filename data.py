import pandas as pd

df_train = pd.read_parquet(
    r"C:\Users\dsng3\Documents\GitHub\DoF_Experiment\data\imdb\train-00000-of-00001.parquet"
)
df_test = pd.read_parquet(
    r"C:\Users\dsng3\Documents\GitHub\DoF_Experiment\data\imdb\test-00000-of-00001.parquet"
)

df_essay_train = pd.read_csv(
    r"C:\Users\dsng3\Documents\GitHub\DoF_Experiment\data\essay\training_set_rel3.tsv",
    sep="\t",
    encoding="latin1"
)

df_essay_valid = pd.read_csv(
    r"C:\Users\dsng3\Documents\GitHub\DoF_Experiment\data\essay\valid_set.tsv",
    sep="\t",
    encoding="latin1"
)

def show_sentence_by_index(dataset: str, idx: int):
    if dataset == "imdb_train":
        if idx in df_train.index:
            print(f"[IMDB Train] idx={idx}:")
            print(df_train.loc[idx, "text"], "\n")
        else:
            print(f"Index {idx} not found in IMDB Train dataset.")
            
    elif dataset == "imdb_test":
        if idx in df_test.index:
            print(f"[IMDB Test] idx={idx}:")
            print(df_test.loc[idx, "text"], "\n")
        else:
            print(f"Index {idx} not found in IMDB Test dataset.")
    
    elif dataset == "essay_train":
        if idx in df_essay_train.index:
            row = df_essay_train.loc[idx]
            print(f"[Essay Train] essay_id={row['essay_id']} | essay_set={row['essay_set']}")
            print(row["essay"], "\n")
        else:
            print(f"Index {idx} not found in Essay Train dataset.")
    
    elif dataset == "essay_valid":
        if idx in df_essay_valid.index:
            row = df_essay_valid.loc[idx]
            print(f"[Essay Valid] essay_id={row['essay_id']} | essay_set={row['essay_set']}")
            print(row["essay"], "\n")
        else:
            print(f"Index {idx} not found in Essay Valid dataset.")
    
    else:
        print("Invalid dataset. Choose from 'imdb_train', 'imdb_test', 'essay_train', 'essay_valid'.")

show_sentence_by_index("imdb_train", 34)
show_sentence_by_index("imdb_test", 100)
show_sentence_by_index("essay_train", 0)   
show_sentence_by_index("essay_valid", 0)   
