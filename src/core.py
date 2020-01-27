import numpy as np
from .stat_funcs import Statistics

mul = np.multiply


class FactorisationMachine:
    def __init__(self, k=3):
        self.v = None
        self.w0 = None  # is global bias
        self.w = None
        self.k = k
        self.X = None
        self.y = None
        self.minibatch_num = 5
        self.verbose = False
        self.statistics = Statistics()

    def fit(self,
            data_shape,  # don't like this decision
            data_generator,
            n_epoch,
            lr=0.001
            ):
        self.w0 = 0.5
        self.w = np.full((data_shape[1], 1), 0.5)
        self.v = np.full((data_shape[1], self.k), 0.5)

        data = list(data_generator)
        for epoch in range(n_epoch):
            print(f"EPOCH : {epoch}")
            mse, r2 = 0, 0
            batch_size = 0
            for i, batch in enumerate(data):
                self.__fit(X=batch[0], y=batch[1],  lr=lr)
                _stat = self.get_statistics()
                mse += _stat[0]
                r2 += _stat[1]
                # batch_size = batch[0].shape[0]
            avg_mse = mse / i
            avg_r2 = r2 / i
            print(f"Average R2 :{avg_r2}\nAverage RMSE : {np.sqrt(avg_mse)} ")

    def __fit(self, X, y, lr=0.001):
        self.X = X
        self.y = y

        self.__minibatch_gd(lr)

    def predict(self, X):
        y = self.__fm_implementation(X)
        return y

    def get_statistics(self):
        y_pred = self.__fm_implementation(self.X)
        err = y_pred - self.y
        r2, mse = self.statistics.get_statistics(self.y, y_pred)
        return mse, r2

    def __minibatch_gd(self, learning_rate):
        """

        :param learning_rate:
        :return:
        """
        mini_batches = self.__create_minibatch(self.X, self.y)
        for x_mb, y_mb in mini_batches:
            # x_mb, y_mb = mini_batch[0], mini_batch[1]
            n = x_mb.shape[0]
            y_pred = self.__fm_implementation(x_mb)
            error = y_mb - y_pred
            lin_p = 2 * learning_rate / n

            # Don't like this part. Some frameworks do it in a different way..
            self.v += lin_p * (x_mb.T.dot(mul(error, (x_mb.dot( self.v)))) -
                               mul(self.v, x_mb.T.power(2).dot(error)))  # WHO AM I, FATHER?
            # self.w += lin_p * np.dot(np.transpose(x_mb), error)
            self.w += lin_p * x_mb.T.dot(error)
            self.w0 += lin_p * np.sum(error)

        if self.verbose:
            self.__verbosity(self.X, self.y)

    def __create_minibatch(self, X, y):
        """
        Using slice as a reason that sparse matrix does not support split operation
        :param X:
        :param y:
        :return:
        """
        batch_size = X.shape[0] // self.minibatch_num
        X_mb = (X[i * batch_size: (i + 1) * batch_size, :] for i in (range(self.minibatch_num)))
        y_mb = (y[i * batch_size: (i + 1) * batch_size, :] for i in (range(self.minibatch_num)))
        return zip(X_mb, y_mb)

    def __fm_implementation(self, X):
        """
        Implementation of 2-way Factorization machine
        interactions: 1/2 * sum(( <X,V.T> ^ 2 ) - < X^2, V^2.T >)
        :param X:
        :return:
        """
        interactions = 0.5 * np.sum(((X.dot(self.v)) ** 2)
                                    - (X.power(2).dot(self.v ** 2)), axis=1)
        y_hat = self.w0 + X.dot(self.w) + interactions.reshape(-1, 1)

        return y_hat

    def __verbosity(self, X, y):
        """
        Just some simple function for verbose output
        :param X:
        :param y:
        :return:
        """

        y_pred = self.__fm_implementation(X)
        err = y_pred - y
        # loss = np.mean(err ** 2)
        # print("Error is {}".format(err))
        _stat = self.statistics.get_statistics(y, y_pred)
        print("MSE is :{}".format(_stat[1]))
        print("R2 is :{}".format(_stat[0]))