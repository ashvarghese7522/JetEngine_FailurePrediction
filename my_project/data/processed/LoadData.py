
import pandas as pd
import matplotlib.pyplot as plt
# Define the file path
column_names = [
    "engine_id", "cycle", "operational_setting_1", "operational_setting_2", "operational_setting_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5", "sensor_6",
    "sensor_7", "sensor_8", "sensor_9", "sensor_10", "sensor_11", "sensor_12",
    "sensor_13", "sensor_14", "sensor_15", "sensor_16", "sensor_17", "sensor_18",
    "sensor_19", "sensor_20", "sensor_21"
]
file_path = r"D:\ProjectML\data\train_FD001.txt"

#Load the dataset
df = pd.read_csv(file_path, delim_whitespace=True, header=None)
#Remove empty columns extra spaces creates empty columns
df=df.dropna(axis=1,how='all')
df.columns = column_names
# Display basic information
print(df.head())
print(df.info())
print(df.describe())


# Filter data for one engine
engine_1 = df[df["engine_id"] == 1]

# Plot a few sensor readings over cycles
plt.figure(figsize=(12, 6))
for sensor in ["sensor_2", "sensor_3", "sensor_4", "sensor_5"]:
    plt.plot(engine_1["cycle"], engine_1[sensor], label=sensor)

plt.xlabel("Cycle")
plt.ylabel("Sensor Readings")
plt.title("Sensor Readings Over Time for Engine 1")
plt.legend()
plt.show()