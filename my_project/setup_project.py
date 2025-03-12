
import os

# Define project name
project_name = "my_project"

# Define folder structure
folders = [
    f"{project_name}/data/raw",
    f"{project_name}/data/processed",
    f"{project_name}/notebooks",
    f"{project_name}/src",
    f"{project_name}/scripts",
    f"{project_name}/models",
    f"{project_name}/reports/figures",
    f"{project_name}/tests",
    f"{project_name}/config"
]

# Define files to create
files = {
    f"{project_name}/README.md": "# Project Title\n\n## Description\n\nAdd project details here.",
    f"{project_name}/requirements.txt": "# List dependencies here",
    f"{project_name}/.gitignore": "models/\n*.pyc\n__pycache__/\n*.pkl\n*.h5\n*.log",
    f"{project_name}/config/config.yaml": "# Configuration settings",
    f"{project_name}/src/data_loader.py": "# Code for loading data",
    f"{project_name}/src/preprocessing.py": "# Code for data preprocessing",
    f"{project_name}/src/feature_engineering.py": "# Code for feature engineering",
    f"{project_name}/src/model.py": "# Code for model definition and training",
    f"{project_name}/src/evaluate.py": "# Code for model evaluation",
    f"{project_name}/src/predict.py": "# Code for making predictions",
    f"{project_name}/scripts/train.py": "# Script for training model",
    f"{project_name}/scripts/inference.py": "# Script for inference",
    f"{project_name}/tests/test_data_loader.py": "# Test cases for data_loader",
    f"{project_name}/tests/test_model.py": "# Test cases for model",
}

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create files with initial content
for file_path, content in files.items():
    with open(file_path, "w") as file:
        file.write(content)

print(f"Project '{project_name}' structure created successfully!")
