from src.utils import DataProcessing


if __name__ == "__main__":
    data_path = "../data/combined_data_1.txt"
    data_processor = DataProcessing(data_path)
    data_processor.preprocess_data(0)
