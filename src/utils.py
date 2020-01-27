import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn import utils
import scipy.sparse
import scipy


class DataProcessing:
    def __init__(self, data_path):
        self.data_path = data_path
        self.main_df = None
        self.encoder = OneHotEncoder(categories='auto')
        self.X = None
        self.y = None

    def get_batch(self, batch_size, X, y):
        for i in range(0, y.shape[0], batch_size):
            # X = self.X[i:i + batch_size, :]
            X_b = X[i:i + batch_size, :]
            y_b = y[i:i + batch_size, :]

            yield X_b, y_b

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

        self.__encode_data()

    def save_data(self, fname):
        self.main_df.to_csv(fname)

    def load_data(self, fname):
        self.main_df = pd.read_csv(fname, index_col=0).drop_duplicates()
        self.__encode_data()
        self.main_df = None  # cleaning the mess
        # self.main_df_arr = np.array_split(self.main_df, self.num_batches)

    def get_data_shape(self):
        return self.X.shape

    def get_fold(self, n_folds=5):
        kf = KFold(n_splits=n_folds)
        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X.tocsr()[train_index], self.X.tocsr()[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            yield (X_train, X_test), (y_train, y_test)

    def __encode_data(self):
        usr_mat = self.encoder.fit_transform(np.asarray(self.main_df["User_ID"]).reshape(-1, 1))
        mov_mat = self.encoder.fit_transform(np.asarray(self.main_df["Movie_ID"]).reshape(-1, 1))
        date_mat = self.encoder.fit_transform(np.asarray(self.main_df["Date"]).reshape(-1, 1))
        self.X = scipy.sparse.hstack([usr_mat, mov_mat, date_mat])
        self.y = np.asarray(self.main_df['Rating']).reshape(-1, 1)  # does not encode actually


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
        end_df = pd.DataFrame(np.array([["0:", None, None]]),
                              columns=['User_ID', "Rating", "Date"])  # auxiliary appendix
        dframe_aux = self.main_df.append(end_df, ignore_index=True)
        null_rat = [i + 1 for i, frame in enumerate(pd.isnull(dframe_aux.Rating)) if frame]

        print(self.main_df.shape)
        print(len(null_rat))

        for i, curr_val in enumerate(null_rat):
            try:
                temp_movies_col = np.append(temp_movies_col, np.full((1, null_rat[i + 1] - curr_val - 1), i + 1))
            except:
                break

        movies_col = pd.DataFrame(temp_movies_col, columns=["Movie_ID"])
        return movies_col
