import pandas as pd


def preprocess_data(data_path):
    comb_df_1 = pd.read_csv(data_path, names=["User_ID", "Rating", "Date"])
    print(comb_df_1.shape)


    return 0
