import numpy as np
import pandas as pd
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

    metrics = Metrics()

    # KNN: Euclidean Predict
    conf_matrix_1 = metrics.confusion_matrix(
        y_predict_euclidean, y_test, "confusion_matrix_euclidean", "Greens"
    )
    accuracy_val_1 = metrics.accuracy(conf_matrix_1)
    precision_val_1 = metrics.f1_score(conf_matrix_1)
    f1_val_1 = metrics.precision(conf_matrix_1)

    # KNN: Manhatthan Predict
    conf_matrix_2 = metrics.confusion_matrix(
        y_predict_manhatthan, y_test, "confusion_matrix_manhatthan", "Blues"
    )
    accuracy_val_2 = metrics.accuracy(conf_matrix_2)
    precision_val_2 = metrics.f1_score(conf_matrix_2)
    f1_val_2 = metrics.precision(conf_matrix_2)

    # Result Table
    table_data = {
        "Model": ["KNN - Euclidean", "KNN - Manhattan"],
        "Accuracy": [accuracy_val_1, accuracy_val_2],
        "Precision": [precision_val_1, precision_val_2],
        "F1-Score": [f1_val_1, f1_val_2],
    }
    df_metrics = pd.DataFrame(table_data)
    print("Metrics Table (Pandas DataFrame):\n")
    print(df_metrics)


if __name__ == "__main__":
    main()
