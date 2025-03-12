import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import Button
from sklearn.preprocessing import MinMaxScaler


class EngineFailureDataset(Dataset):
    def __init__(self, file_path):
        """
        Load and preprocess the dataset.
        """
        self.data = self.load_data(file_path)
        self.constant_columns = self.identify_constant_columns()
        self.data = compute_rul(self.data)
        self.data = normalize_data(self.data)

    def load_data(self, file_path):
        """
        Read CSV file and assign appropriate column names.
        """
        data = pd.read_csv(file_path, sep='\s+', header=None)
        data.columns = ["unit_number", "time_in_cycles", "op_setting_1", "op_setting_2", "op_setting_3"] + \
                       [f"sensor_{i}" for i in range(1, 22)]
        return data.astype(float)  # Convert all values to float

    def identify_constant_columns(self):
        """
        Identify columns with constant values.
        """
        return [col for col in self.data.columns if self.data[col].nunique() == 1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data.iloc[idx].values, dtype=torch.float32)


def compute_rul(data):
    """
    Compute Remaining Useful Life (RUL) for each engine.
    """
    max_cycles = data.groupby("unit_number")["time_in_cycles"].max()
    data["RUL"] = data["unit_number"].map(max_cycles) - data["time_in_cycles"]
    return data


def normalize_data(data):
    """
    Normalize sensor data using MinMaxScaler.
    """
    scaler = MinMaxScaler()
    cols_to_normalize = data.columns[2:]  # Exclude 'unit_number' and 'time_in_cycles'
    data[cols_to_normalize] = scaler.fit_transform(data[cols_to_normalize])
    return data


def load_dataloader(file_path, batch_size=64, shuffle=True):
    """
    Initialize dataset and DataLoader.
    """
    dataset = EngineFailureDataset(file_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset, dataloader


def explore_batch(dataloader, dataset):
    """
    Fetch a single batch and display key statistics.
    """
    for batch in dataloader:
        batch_df = pd.DataFrame(batch.numpy(), columns=dataset.data.columns)
        print("Head:\n", batch_df.head())
        print("Tail:\n", batch_df.tail())
        print("Dtype:\n", batch_df.dtypes)
        print("Description:\n", batch_df.describe())
        print("Batch Shape:", batch.shape)
        print("Batch Mean:\n", batch_df.mean())
        print("Batch Std Dev:\n", batch_df.std())
        print("Column Names:", dataset.data.columns.tolist())
        print("Constant Columns:", dataset.constant_columns)
        print("Number of Duplicate Rows:", dataset.data.duplicated().sum())
        print("Missing Values:\n", dataset.data.isnull().sum())
        dataset.data.fillna(method="ffill", inplace=True)  # Forward fill missing values

        failed_engines = dataset.data.groupby("unit_number")["time_in_cycles"].max()
        dataset.data["is_last_50_cycles"] = dataset.data.apply(
            lambda x: 1 if x["time_in_cycles"] > (failed_engines[x["unit_number"]] - 50) else 0, axis=1
        )
        sns.boxplot(x="is_last_50_cycles", y="sensor_2", data=dataset.data)
        plt.title("Sensor 2 Behavior Before Failure")
        plt.show()

        break  # Process only one batch


def plot_sensor_trends(dataset):
    """
    Interactive sensor trend visualization for engines.
    """
    engine_ids = dataset.data["unit_number"].unique()
    index = [0]  # Mutable index for button interaction

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(bottom=0.3)

    ax_next = plt.axes([0.8, 0.05, 0.1, 0.075])
    ax_prev = plt.axes([0.65, 0.05, 0.1, 0.075])
    button_next = Button(ax_next, 'Next')
    button_prev = Button(ax_prev, 'Previous')

    def plot_current_engine():
        ax.clear()
        engine_id = engine_ids[index[0]]
        engine_data = dataset.data[dataset.data["unit_number"] == engine_id]

        for i in range(1, 5):  # Plot first 4 sensors
            ax.plot(engine_data["time_in_cycles"], engine_data[f"sensor_{i}"], label=f"Sensor {i}")

        ax.set_xlabel("Time in Cycles")
        ax.set_ylabel("Sensor Readings")
        ax.legend()
        ax.set_title(f"Sensor Trends for Engine {engine_id}")
        plt.draw()

    def next_engine(event):
        if index[0] < len(engine_ids) - 1:
            index[0] += 1
            plot_current_engine()

    def prev_engine(event):
        if index[0] > 0:
            index[0] -= 1
            plot_current_engine()

    button_next.on_clicked(next_engine)
    button_prev.on_clicked(prev_engine)

    plot_current_engine()
    plt.show(block=True)


def visualize_data(dataset):
    """
    Interactive sensor data distribution visualization.
    """
    columns = dataset.data.columns[5:]  # Exclude non-sensor columns
    index = [0]

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.3)

    ax_next = plt.axes([0.8, 0.05, 0.1, 0.075])
    ax_prev = plt.axes([0.65, 0.05, 0.1, 0.075])
    button_next = Button(ax_next, 'Next')
    button_prev = Button(ax_prev, 'Previous')

    def plot_sensor():
        ax.clear()
        sns.histplot(dataset.data[columns[index[0]]], bins=50, kde=True, ax=ax)
        ax.set_title(f"Distribution of {columns[index[0]]}")
        plt.draw()

    def next_sensor(event):
        if index[0] < len(columns) - 1:
            index[0] += 1
            plot_sensor()

    def prev_sensor(event):
        if index[0] > 0:
            index[0] -= 1
            plot_sensor()

    button_next.on_clicked(next_sensor)
    button_prev.on_clicked(prev_sensor)

    plot_sensor()
    plt.show(block=True)


if __name__ == "__main__":
    file_path = "D:/ProjectML/Engine_Failure/my_project/data/raw/train_FD001.txt"
    dataset, dataloader = load_dataloader(file_path)

    # Run Exploratory Analysis
    explore_batch(dataloader, dataset)

    # Interactive Visualizations
    plot_sensor_trends(dataset)
    visualize_data(dataset)
