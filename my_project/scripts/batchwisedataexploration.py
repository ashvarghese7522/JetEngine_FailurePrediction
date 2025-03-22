import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


class EngineFailureDataset(Dataset):
    def __init__(self, train_file, test_file, rul_file=None, train=True, scaler=None, save_path=None):
        self.train = train
        self.data = self.load_data(train_file if train else test_file)

        # Remove constant columns
        self.constant_columns = self.identify_constant_columns()
        self.data.drop(columns=self.constant_columns, inplace=True)

        # Handle missing values
        self.data.fillna(self.data.median(), inplace=True)

        # Remove highly correlated features
        self.data = self.remove_highly_correlated_features()

        # Compute or merge RUL
        if train:
            self.data = self.compute_rul()
        else:
            self.data = self.merge_test_rul(rul_file)

        # Normalize data (pass scaler to avoid leakage)
        self.scaler = scaler if scaler else MinMaxScaler()
        self.data = self.normalize_data()

        # Save cleaned data if save_path is provided
        if save_path:
            self.save_cleaned_data(save_path)

        # Debugging output
        self.debug_dataset()

    def load_data(self, file_path):
        """Load dataset and assign column names"""
        data = pd.read_csv(file_path, delim_whitespace=True, header=None)
        data.columns = ["unit_number", "time_in_cycles", "op_setting_1", "op_setting_2", "op_setting_3"] + \
                       [f"sensor_{i}" for i in range(1, 22)]
        return data.astype(float)

    def identify_constant_columns(self):
        """Identify and return constant columns"""
        return [col for col in self.data.columns if self.data[col].nunique() == 1]

    def remove_highly_correlated_features(self, threshold=0.9):
        """Remove highly correlated features"""
        corr_matrix = self.data.iloc[:, 5:].corr(method='spearman').abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        return self.data.drop(columns=to_drop, axis=1)

    def compute_rul(self):
        """Compute Remaining Useful Life (RUL)"""
        max_cycles = self.data.groupby("unit_number")["time_in_cycles"].max()
        self.data["RUL"] = self.data["unit_number"].map(max_cycles) - self.data["time_in_cycles"]
        return self.data

    def merge_test_rul(self, rul_file):
        """Merge actual RUL values into the test dataset"""
        rul = pd.read_csv(rul_file, delim_whitespace=True, header=None, names=["RUL"])
        last_cycles = self.data.groupby("unit_number")["time_in_cycles"].max().reset_index()
        last_cycles.sort_values("unit_number", inplace=True)
        rul = rul.reset_index(drop=True)

        if len(last_cycles) != len(rul):
            raise ValueError(f"Mismatch: {len(last_cycles)} last cycles vs {len(rul)} RUL values")

        last_cycles["RUL"] = rul["RUL"].values
        self.data = self.data.merge(last_cycles[["unit_number", "RUL"]], on="unit_number", how="left")
        return self.data

    def normalize_data(self):
        """Normalize sensor and operational data"""
        cols_to_normalize = [col for col in self.data.columns if col not in ["unit_number", "time_in_cycles", "RUL"]]
        if self.train:
            self.scaler.fit(self.data[cols_to_normalize])
        self.data[cols_to_normalize] = self.scaler.transform(self.data[cols_to_normalize])
        return self.data

    def save_cleaned_data(self, save_path):
        """Save cleaned dataset to specified path"""
        file_name = "train_cleaned.csv" if self.train else "test_cleaned.csv"
        full_path = f"{save_path}/{file_name}"
        self.data.to_csv(full_path, index=False)
        print(f"Cleaned data saved to {full_path}")

    def debug_dataset(self):
        """Print dataset details for debugging"""
        print("Dataset Shape:", self.data.shape)
        print("\nDataset Head:\n", self.data.head())
        print("\nMissing Values:\n", self.data.isnull().sum().sum())
        print("\nConstant Columns Removed:", self.constant_columns)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Return features (X) and target (y) as tensors"""
        row = self.data.iloc[idx]
        features = row.drop(["unit_number", "time_in_cycles", "RUL"]).values
        label = row["RUL"]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


if __name__ == "__main__":
    train_file = "D:/ProjectML/Engine_Failure/my_project/data/raw/train_FD001.txt"
    test_file = "D:/ProjectML/Engine_Failure/my_project/data/raw/test_FD001.txt"
    rul_file = "D:/ProjectML/Engine_Failure/my_project/data/raw/RUL_FD001.txt"
    save_path = "D:/ProjectML/Engine_Failure/my_project/data/processed"

    print("Loading Train Dataset...")
    train_scaler = MinMaxScaler()
    train_dataset = EngineFailureDataset(train_file, test_file, train=True, scaler=train_scaler, save_path=save_path)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)

    print("Loading Test Dataset...")
    test_dataset = EngineFailureDataset(train_file, test_file, rul_file, train=False, scaler=train_scaler,
                                        save_path=save_path)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=True)

    print("\nTrain Data Sample (RUL Check):")
    print(train_dataset.data[["unit_number", "time_in_cycles", "RUL"]].head(10))

    print("\nTest Data Sample (RUL Check):")
    print(test_dataset.data[["unit_number", "time_in_cycles", "RUL"]].tail(10))