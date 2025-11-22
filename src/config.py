# CONFIG.PY
# Contains global configuration values

DATA_PATH = "D:/bank_loan_ml/data/bankloan.csv"   # <-- change this if file name differs
TARGET = "Personal.Loan"      # numeric 0/1 target

TEST_SIZE = 0.2               # 80/20 split
RANDOM_STATE = 42             # reproducibility

# Models we want to train
MODELS = ["linear", "knn"]
