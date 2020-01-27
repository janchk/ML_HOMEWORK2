from __future__ import print_function

from utils import DataProcessing
from core import FactorisationMachine

import os

if __name__ == "__main__":
    data_path = "../data/combined_data_1.txt"
    data_processor = DataProcessing(data_path)
    batch_size = 100000

    processed_fname = "processed_data.csv"
    if not os.path.exists(processed_fname):
        data_processor.preprocess_data()
        data_processor.save_data('processed_data.csv')
    else:
        data_processor.load_data(processed_fname)
    fm = FactorisationMachine()

    data_shape = data_processor.get_data_shape()

    for f_num, fold in enumerate(data_processor.get_fold(5)):
        print("fold number {}".format(f_num))

        (X_train, X_test), (y_train, y_test) = fold
        fm.fit(data_shape, data_processor.get_batch(batch_size, X_train, y_train), 10)


        # Predict time
        y_pred = fm.predict(X_test)
        y_gt = y_test
        r2, rmse = fm.statistics.get_statistics(y_gt, y_pred)
        print("TEST: RMSE is: {}".format(rmse))
