from src.load_data import load_dataset
from src.preprocess import prepare_classification, prepare_regression
from src.train_models import LinearRegression, KNNRegressor
from src.evaluate import evaluate_classification_from_regressor, evaluate_regression
from src.config import K_FOR_KNN, THRESHOLD, DATA_PATH

def main():
    data_path = DATA_PATH
    print("Loading data...")
    df = load_dataset(data_path)
    print("Dataset shape:", df.shape)

    Xc_train, Xc_test, yc_train, yc_test, scaler_c, feats_c = prepare_classification(df)
    print("Classification features:", feats_c)

    lin_clf = LinearRegression()
    lin_clf.fit(Xc_train, yc_train)
    print("Evaluating Linear (as classifier by threshold)...")
    evaluate_classification_from_regressor("LinearReg (thresholded)", lin_clf, Xc_test, yc_test, threshold=THRESHOLD)

    knn_clf = KNNRegressor(k=K_FOR_KNN)
    knn_clf.fit(Xc_train, yc_train)
    print("Evaluating KNN (as classifier by threshold)...")
    evaluate_classification_from_regressor("KNNReg (thresholded)", knn_clf, Xc_test, yc_test, threshold=THRESHOLD)

    Xr_train, Xr_test, yr_train, yr_test, scaler_r, feats_r = prepare_regression(df)
    print("Regression features:", feats_r)

    lin_reg = LinearRegression()
    lin_reg.fit(Xr_train, yr_train)
    print("Evaluating Linear Regression...")
    evaluate_regression("LinearReg ", lin_reg, Xr_test, yr_test)

    knn_reg = KNNRegressor(k=K_FOR_KNN)
    knn_reg.fit(Xr_train, yr_train)
    print("Evaluating KNN Regressor...")
    evaluate_regression("KNNReg ", knn_reg, Xr_test, yr_test)

if __name__ == "__main__":
    main()
