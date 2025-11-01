import numpy as np
from loguru import logger

from train.hold_out import HoldOut
from train.knn import KNN
from train.metrics_graph import Metrics


def read_data():
    logger.info("[read_data] Reading txt")

    try:
        data = np.loadtxt("data/data_banknote_authentication.txt", delimiter=",")
        logger.info(f"[read_data] Data shape (lines and columns): {data.shape}")
    
        return data
    except Exception as e:
        logger.error(f"[read_data] ERROR: {e}")


def main():
    data = read_data()

    x = data[:, :-1]
    y = data[:, -1]
    
    hold_out = HoldOut(x=x, y=y)
    
    x_train, x_test, y_train, y_test = hold_out.execute_method()
    knn = KNN(k=3, x_train=x_train, y_train=y_train, task="C")
    
    y_predict_euclidean, y_predict_manhatthan = knn.predict(x_test=x_test)

    Metrics.confusion_matrix(y_predict_euclidean, y_test, "confusion_matrix_euclidean", "Greens")
    Metrics.confusion_matrix(y_predict_manhatthan, y_test, "confusion_matrix_manhatthan", "Blues")

if __name__ == "__main__":
    main()
