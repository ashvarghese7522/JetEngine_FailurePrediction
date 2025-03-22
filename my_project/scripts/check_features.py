import pandas as pd

# Load the processed dataset
df = pd.read_csv("D:/ProjectML/Engine_Failure/my_project/data/processed/train_features.csv")

# Display first 5 rows
print(df.head())

# Show column names to verify new features
print("\nColumn Names:\n", df.columns.tolist())

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum().sum())

# Summary statistics of rolling & lag features
rolling_cols = [col for col in df.columns if "rolling" in col or "lag" in col]
print("\nRolling & Lag Features Summary:\n", df[rolling_cols].describe())
