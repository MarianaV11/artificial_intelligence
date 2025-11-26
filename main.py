import time

import numpy as np
import pandas as pd
from loguru import logger
from scipy.io import arff

from train.bayes_multivariado import BayesMultivariado
from train.bayes_univariado import BayesUnivariado
from train.perceptron import Perceptron
from train.mlp import MLP
from train.k_fold import KFoldCV
from train.knn import KNN
from train.metrics_graph import Metrics


def read_data():
    data, meta = arff.loadarff("data/dataset_28.arff")

    return np.array(data.tolist(), dtype=float)


def main():
    data = read_data()
    x = data[:, :-1]
    y = data[:, -1]

    kfold = KFoldCV(x=x, y=y, n_splits=10)
    metrics = Metrics()

    results_summary = {
        "Model": ["KNN - Euclidean", "KNN - Manhattan", "Bayes - Univariado", "Bayes - Multivariado", "Perceptron", "MLP"],
        "Accuracy": [0, 0, 0, 0, 0, 0],
        "Precision": [0, 0, 0, 0, 0, 0],
        "F1-Score": [0, 0, 0, 0, 0, 0],
        "Train Time (s)": [0, 0, 0, 0, 0, 0],
        "Test Time (s)": [0, 0, 0, 0, 0, 0],
    }

    folds = kfold.execute_method()
    logger.info(f"[main] Executando {len(folds)} folds...")

    for fold_idx, (x_train, x_test, y_train, y_test) in enumerate(folds, start=1):
        logger.info(f"[main] Fold {fold_idx} iniciado...")

        # --- KNN ---
        start_train = time.time()
        knn = KNN(k=3, x_train=x_train, y_train=y_train, task="C")
        train_time = time.time() - start_train

        start_test = time.time()
        y_pred_euclid, y_pred_manhattan = knn.predict(x_test=x_test)
        test_time = time.time() - start_test

        conf_euc = metrics.confusion_matrix(y_pred_euclid, y_test, f"fold{fold_idx}_knn_euclidean", "Greens")
        conf_man = metrics.confusion_matrix(y_pred_manhattan, y_test, f"fold{fold_idx}_knn_manhattan", "Blues")

        acc_euc = metrics.accuracy(conf_euc)
        prec_euc = metrics.precision(conf_euc)
        f1_euc = metrics.f1_score(conf_euc)

        acc_man = metrics.accuracy(conf_man)
        prec_man = metrics.precision(conf_man)
        f1_man = metrics.f1_score(conf_man)

        results_summary["Accuracy"][0] += acc_euc
        results_summary["Precision"][0] += prec_euc
        results_summary["F1-Score"][0] += f1_euc
        results_summary["Train Time (s)"][0] += train_time
        results_summary["Test Time (s)"][0] += test_time

        results_summary["Accuracy"][1] += acc_man
        results_summary["Precision"][1] += prec_man
        results_summary["F1-Score"][1] += f1_man
        results_summary["Train Time (s)"][1] += train_time
        results_summary["Test Time (s)"][1] += test_time

        # --- Bayes Univariado ---
        start_train = time.time()
        bayes_uni = BayesUnivariado(x_train=x_train, y_train=y_train)
        train_time = time.time() - start_train

        start_test = time.time()
        y_pred_uni = bayes_uni.predict(x_test=x_test)
        test_time = time.time() - start_test

        conf_uni = metrics.confusion_matrix(y_pred_uni, y_test, f"fold{fold_idx}_bayes_uni", "Oranges")
        acc_uni = metrics.accuracy(conf_uni)
        prec_uni = metrics.precision(conf_uni)
        f1_uni = metrics.f1_score(conf_uni)

        results_summary["Accuracy"][2] += acc_uni
        results_summary["Precision"][2] += prec_uni
        results_summary["F1-Score"][2] += f1_uni
        results_summary["Train Time (s)"][2] += train_time
        results_summary["Test Time (s)"][2] += test_time

        # --- Bayes Multivariado ---
        start_train = time.time()
        bayes_multi = BayesMultivariado(x_train=x_train, y_train=y_train)
        train_time = time.time() - start_train

        start_test = time.time()
        y_pred_multi = bayes_multi.predict(x_test=x_test)
        test_time = time.time() - start_test

        conf_multi = metrics.confusion_matrix(y_pred_multi, y_test, f"fold{fold_idx}_bayes_multi", "Purples")
        acc_multi = metrics.accuracy(conf_multi)
        prec_multi = metrics.precision(conf_multi)
        f1_multi = metrics.f1_score(conf_multi)

        results_summary["Accuracy"][3] += acc_multi
        results_summary["Precision"][3] += prec_multi
        results_summary["F1-Score"][3] += f1_multi
        results_summary["Train Time (s)"][3] += train_time
        results_summary["Test Time (s)"][3] += test_time

        # --- Perceptron ---
        start_train = time.time()
        perceptron = Perceptron(x=x_train, y=y_train)
        train_time = time.time() - start_train

        start_test = time.time()
        y_perceptron = perceptron.predict(x_test=x_test)
        test_time = time.time() - start_test

        conf_perceptron = metrics.confusion_matrix(y_perceptron, y_test, f"fold{fold_idx}_bayes_multi", "Purples")
        acc_perceptron = metrics.accuracy(conf_perceptron)
        prec_perceptron = metrics.precision(conf_perceptron)
        f1_perceptron = metrics.f1_score(conf_perceptron)

        results_summary["Accuracy"][4] += acc_perceptron
        results_summary["Precision"][4] += prec_perceptron
        results_summary["F1-Score"][4] += f1_perceptron
        results_summary["Train Time (s)"][4] += train_time
        results_summary["Test Time (s)"][4] += test_time

        logger.info(f"[main] Fold {fold_idx} concluído.")

        # --- MLP ---
        start_train = time.time()
        mlp = MLP(input_dim=x_test.shape[1])
        train_time = time.time() - start_train

        start_test = time.time()
        y_mlp = mlp.predict(X=x_test)
        test_time = time.time() - start_test

        conf_mlp = metrics.confusion_matrix(y_mlp, y_test, f"fold{fold_idx}_bayes_multi", "Purples")
        acc_mlp = metrics.accuracy(conf_mlp)
        prec_mlp = metrics.precision(conf_mlp)
        f1_mlp = metrics.f1_score(conf_mlp)

        results_summary["Accuracy"][5] += acc_mlp
        results_summary["Precision"][5] += prec_mlp
        results_summary["F1-Score"][5] += f1_mlp
        results_summary["Train Time (s)"][5] += train_time
        results_summary["Test Time (s)"][5] += test_time

        logger.info(f"[main] Fold {fold_idx} concluído.")

    # --- Calcula médias finais ---
    num_folds = len(folds)
    for i in range(len(results_summary["Model"])):
        results_summary["Accuracy"][i] /= num_folds
        results_summary["Precision"][i] /= num_folds
        results_summary["F1-Score"][i] /= num_folds
        results_summary["Train Time (s)"][i] /= num_folds
        results_summary["Test Time (s)"][i] /= num_folds

    df_metrics = pd.DataFrame(results_summary)
    print("\nMédias das Métricas e Tempos (10-Fold Cross-Validation):\n")
    print(df_metrics.round(4))


if __name__ == "__main__":
    main()
