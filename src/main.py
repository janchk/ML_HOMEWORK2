from utils import DataProcessing
from core import FactorisationMachine

import os

if __name__ == "__main__":
    data_path = "../data/combined_data_1.txt"
    data_processor = DataProcessing(data_path, 10)

    processed_fname = "processed_data.csv"
    if not os.path.exists(processed_fname):
        data_processor.preprocess_data()
        data_processor.save_data('processed_data.csv')
    else:
        data_processor.load_data(processed_fname)

    batch = data_processor.get_batch(0)
    fm = FactorisationMachine()
    fm.fit(X=batch[0], y=batch[1])

    pass
