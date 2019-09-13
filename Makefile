# Create env
conda env create -f environment.yaml
conda activate model_fairness
python -m ipykernel install --user --name model_fairness --display-name "model_fairness"