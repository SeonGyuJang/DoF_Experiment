import pandas as pd
df_train = pd.read_parquet(r"C:\Users\dsng3\Documents\GitHub\DoF_Experiment\data\train-00000-of-00001.parquet")
df_valid = pd.read_parquet(r"C:\Users\dsng3\Documents\GitHub\DoF_Experiment\data\validation-00000-of-00001.parquet")
df_test = pd.read_parquet(r"C:\Users\dsng3\Documents\GitHub\DoF_Experiment\data\test-00000-of-00001.parquet")

# Train Data
print("============ Train ============")
print(df_train.head())
print("\nLabel counts in Train data:")
print(df_train['label'].value_counts())
print("==============================")
print("\n")

# Validation Data
print("=========== Validation ===========")
print(df_valid.head())
print("\nLabel counts in Validation data:")
print(df_valid['label'].value_counts())
print("==================================")
print("\n")

# Test Data
print("============ Test ================")
print(df_test.head())
print("\nLabel counts in Test data:")
print(df_test['label'].value_counts())
print("==================================")