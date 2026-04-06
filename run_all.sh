#!/bin/bash
# 1. Setup Environment
echo "Cleaning old results..."
rm -rf results && mkdir results
rm -rf data && mkdir data

# 2. Dependencies
echo "Installing Python dependencies..."
pip install numpy pandas matplotlib scikit-learn ripser gudhi cvxpy tqdm

# 3. Data Check (User needs to provide zoo.csv)
if [ ! -f "data/zoo.csv" ]; then
    echo "Warning: data/zoo.csv not found. Running synthetic tasks only."
fi

# 4. Run Tasks
echo "Running clustering pipeline (10 methods)..."
python3 tasks/main_pipeline.py

echo "Execution finished. Charts generated in results/ directory."
