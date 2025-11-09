import numpy as np
import pandas as pd
from loguru import logger

from train.k_fold import KFoldCV
from train.knn import KNN
from train.metrics_graph import Metrics
from train.bayes_univariado import BayesUnivariado
from train.bayes_multivariado import BayesMultivariado


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

    kfold = KFoldCV(x=x, y=y, n_splits=10)
    metrics = Metrics()

    results_summary = {
        "Model": ["KNN - Euclidean", "KNN - Manhattan", "Bayes - Univariado", "Bayes - Multivariado"],
        "Accuracy": [0, 0, 0, 0],
        "Precision": [0, 0, 0, 0],
        "F1-Score": [0, 0, 0, 0],
    }

    folds = kfold.execute_method()
    logger.info(f"[main] Executando {len(folds)} folds...")

    for fold_idx, (x_train, x_test, y_train, y_test) in enumerate(folds, start=1):
        logger.info(f"[main] Fold {fold_idx} iniciado...")

        knn = KNN(k=3, x_train=x_train, y_train=y_train, task="C")
        y_pred_euclid, y_pred_manhattan = knn.predict(x_test=x_test)

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

        results_summary["Accuracy"][1] += acc_man
        results_summary["Precision"][1] += prec_man
        results_summary["F1-Score"][1] += f1_man

        # --- Bayes Univariado ---
        bayes_uni = BayesUnivariado(x_train=x_train, y_train=y_train)
        y_pred_uni = bayes_uni.predict(x_test=x_test)

        conf_uni = metrics.confusion_matrix(y_pred_uni, y_test, f"fold{fold_idx}_bayes_uni", "Oranges")
        acc_uni = metrics.accuracy(conf_uni)
        prec_uni = metrics.precision(conf_uni)
        f1_uni = metrics.f1_score(conf_uni)

        results_summary["Accuracy"][2] += acc_uni
        results_summary["Precision"][2] += prec_uni
        results_summary["F1-Score"][2] += f1_uni

        # --- Bayes Multivariado ---
        bayes_multi = BayesMultivariado(x_train=x_train, y_train=y_train)
        y_pred_multi = bayes_multi.predict(x_test=x_test)

        conf_multi = metrics.confusion_matrix(y_pred_multi, y_test, f"fold{fold_idx}_bayes_multi", "Purples")
        acc_multi = metrics.accuracy(conf_multi)
        prec_multi = metrics.precision(conf_multi)
        f1_multi = metrics.f1_score(conf_multi)

        results_summary["Accuracy"][3] += acc_multi
        results_summary["Precision"][3] += prec_multi
        results_summary["F1-Score"][3] += f1_multi

        logger.info(f"[main] Fold {fold_idx} concluído.")

    num_folds = len(folds)
    for i in range(len(results_summary["Model"])):
        results_summary["Accuracy"][i] /= num_folds
        results_summary["Precision"][i] /= num_folds
        results_summary["F1-Score"][i] /= num_folds

    df_metrics = pd.DataFrame(results_summary)
    print("\nMédias das Métricas (10-Fold Cross-Validation):\n")
    print(df_metrics)


if __name__ == "__main__":
    main()
