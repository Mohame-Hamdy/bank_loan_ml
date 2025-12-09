
# Contains global configuration values

DATA_PATH = "D:/bank_loan_ml/data/bankloan.csv"   
# Configuration constants
CLASS_TARGET = "Personal.Loan"   # binary target for classification (0/1)
REG_TARGET = "Income"            # continuous target for regression

DROP_COLUMNS = ["ID", "ZIP.Code"]

TEST_SIZE = 0.2
RANDOM_STATE = 42

# KNN hyperparameter for regression and for KNN probability in classification
K_FOR_KNN = 5



K_FOR_KNN = 5       # default k for KNN regressor
THRESHOLD = 0.5  