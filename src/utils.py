import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


class DataProcessing:
    def __init__(self, data_path):
        self.data_path = data_path
        # self.movies_col = np.array([])
        self.num_batches = 5
        self.main_df = None
        # self.kf = KFold(n_splits=self.num_folds)
        # self.data = None

    def get_batch(self, batch_num):
        pivoted_df = pd.pivot_table(self.main_df[batch_num], values="Rating", index="User_ID", columns="Movie_ID")
        return pivoted_df

    def preprocess_data(self):
        self.main_df = pd.read_csv(self.data_path, names=["User_ID", "Rating", "Date"])
        # self.__get_movie_ratings()

        movies_df = self.__get_movie_ratings()

        self.main_df = self.main_df[pd.notnull(self.main_df['Rating'])]
        self.main_df = self.main_df.reset_index(drop=True)
        self.main_df["Movie_ID"] = movies_df

        print('Original Shape: {}'.format(self.main_df.shape))
        self.__clean_data()
        print('After Trim Shape: {}'.format(self.main_df.shape))

        self.main_df = np.array_split(self.main_df, self.num_batches)
        # Pivoting Data
        # pivoted_df = pd.pivot_table(self.main_df, values="Rating", index="User_ID", columns="Movie_ID")


        return 0

    def __clean_data(self):
        f = ['count', 'mean']

        df_movie_summary = self.main_df.groupby('Movie_ID')['Rating'].agg(f)
        df_movie_summary.index = df_movie_summary.index.map(int)
        movie_benchmark = round(df_movie_summary['count'].quantile(0.75), 0)
        drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index

        print('Movie minimum times of review: {}'.format(movie_benchmark))

        df_user_summary = self.main_df.groupby('User_ID')['Rating'].agg(f)
        df_user_summary.index = df_user_summary.index.map(int)
        user_benchmark = round(df_user_summary['count'].quantile(0.75), 0)
        drop_cust_list = df_user_summary[df_user_summary['count'] < user_benchmark].index

        print('Customer minimum times of review: {}'.format(user_benchmark))

        self.main_df = self.main_df[~self.main_df['Movie_ID'].isin(drop_movie_list)]
        self.main_df = self.main_df[~self.main_df['User_ID'].isin(drop_cust_list)]

    def __get_movie_ratings(self):
        temp_movies_col = np.array([])
        end_df = pd.DataFrame(np.array([["0:", None, None]]), columns=['User_ID', "Rating", "Date"])  # auxiliary appendix
        dframe_aux = self.main_df.append(end_df, ignore_index=True)
        null_rat = [i + 1 for i, frame in enumerate(pd.isnull(dframe_aux.Rating)) if frame]

        print(self.main_df.shape)
        print(len(null_rat))

        for i, curr_val in enumerate(null_rat):
            try:
                temp_movies_col = np.append(temp_movies_col, np.full((1, null_rat[i+1] - curr_val - 1), i + 1))
            except:
                break

        movies_col = pd.DataFrame(temp_movies_col, columns=["Movie_ID"])
        return movies_col



