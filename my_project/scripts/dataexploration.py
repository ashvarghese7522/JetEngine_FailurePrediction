import pandas as pd
import os

# Define absolute path
data_dir = "D:\ProjectML\Engine_Failure\my_project/data/raw"
datasets = ["FD001", "FD002", "FD003", "FD004"]
data_dict = {}

# Load all datasets
for dataset in datasets:
    file_path = os.path.join(data_dir, f"train_{dataset}.txt")  # Use absolute path

    try:
        df = pd.read_csv(file_path, sep=" ", header=None).dropna(axis=1, how="all")  # Drop extra spaces
        data_dict[dataset] = df  # Store using correct key
        print(f"✅ Loaded {file_path} successfully!")
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")

# Check if data is loaded before accessing it
#if "FD001" in data_dict:
   # print(data_dict["FD001"].head())
#else:
#    print("❌ FD001 data not loaded. Check file paths.")
print(data_dict["FD001"].shape)
print(data_dict["FD002"].shape)
print(data_dict["FD003"].shape)
print(data_dict["FD004"].shape)
