# MAIN.PY
# Entire pipeline — load → preprocess → train → evaluate

from src.load_data import load_raw_data
from src.preprocess import preprocess_data
from src.train_models import train_all_models
from src.evaluate import evaluate_model

def main():

    print("Loading data...")
    df = load_raw_data()

    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    print("Training models...")
    models = train_all_models(X_train, y_train)

    print("Evaluating models...")
    for name, model in models.items():
        evaluate_model(name, model, X_test, y_test)


if __name__ == "__main__":
    main()
