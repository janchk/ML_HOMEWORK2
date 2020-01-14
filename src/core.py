import numpy as np
from stat_funcs import mse, r2
from sklearn.utils import shuffle

mul = np.multiply


class FactorisationMachine:
    def __init__(self, k=1):
        self.v = None
        self.w0 = None  # is global bias
        self.w = None
        self.k = k
        self.minibatch_num = 5
        self.verbose = True

    def fit(self, X, y, epochs=10, lr=0.001):
        # TODO think about init values
        self.w0 = 0.5
        self.w = np.full((X.shape[1], 1), 0.5)
        self.v = np.full((X.shape[1], self.k), 0.5)
        self.__minibatch_gd(X, y, epochs, lr)

    def predict(self, X):
        y = self.__fm_implementation(X)
        return y

    def __minibatch_gd(self, X, y, epochs, learning_rate):
        """

        :param X:
        :param y:
        :param epochs:
        :param learning_rate:
        :return:
        """
        for epoch in range(epochs):
            mini_batches = self.__create_minibatch(X, y)
            for x_mb, y_mb in mini_batches:
                # x_mb, y_mb = mini_batch[0], mini_batch[1]
                n = x_mb.shape[0]
                y_pred = self.__fm_implementation(x_mb)
                error = y_mb - y_pred
                # log_loss = np.log(np.exp(-y_pred * y_mb) + 1.0)  # log loss
                lin_p = 2 * learning_rate / n

                # Don't like this part. Some frameworks do it in a different way..
                self.v += lin_p * (x_mb.T.dot(mul(error, (x_mb.dot( self.v)))) -
                                   mul(self.v, x_mb.T.power(2).dot(error)))  # WHO AM I, FATHER?
                # self.w += lin_p * np.dot(np.transpose(x_mb), error)
                self.w += lin_p * x_mb.T.dot(error)
                self.w0 += lin_p * np.sum(error)

                if self.verbose:
                    self.__verbosity(X, y, epoch)

    def __create_minibatch(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        batch_size = X.shape[0] // self.minibatch_num
        X_mini = [X[i * batch_size: (i + 1) * batch_size, :] for i in (range(self.minibatch_num))]
        y_mini = [y[i * batch_size: (i + 1) * batch_size, :] for i in (range(self.minibatch_num))]
        return zip(X_mini, y_mini)

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

    def __verbosity(self, X, y, epoch_n):
        """
        Just some simple function for verbose output
        :param X:
        :param y:
        :return:
        """

        y_pred = self.__fm_implementation(X)
        err = y_pred - y
        _mse = mse(y, y_pred)
        _r2 = r2(y, y_pred)

        # print("Error is {}".format(err))
        print("Epoch is {}".format(epoch_n))
        print("MSE is {}".format(_mse))
        print("R2 is {}".format(_r2))
