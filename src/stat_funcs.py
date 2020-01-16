import numpy as np


class Statistics:
    def __init__(self):
        pass

    def mse(self, ground_truth, predictions):
        """

        :param ground_truth: vectors of GT values
        :param predictions: vectors of predicted values
        :return: Mean Square Error
        """
        return np.sum((ground_truth - predictions) ** 2) / len(predictions)

    def r2(self, ground_truth, predictions):
        mean_pred_value = np.mean(ground_truth)
        ss_tot = np.sum((ground_truth - mean_pred_value) ** 2)
        ss_res = np.sum((ground_truth - predictions) ** 2)

        return 1 - (ss_res / ss_tot)

    def get_statistics(self, ground_truth, predictions):
        r2 = self.r2(ground_truth, predictions)
        mse = self.mse(ground_truth, predictions)
        return r2, mse

