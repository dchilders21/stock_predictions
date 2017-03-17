import os
import math, time
import numpy as np
from numpy import newaxis
import pandas as pd
import pandas_datareader

from quote import GoogleIntradayQuote
from datetime import datetime
import model_libs

# 30 min before opening of market
# Run 1 model, Opening - and also gather data for 10 minute model
# Once opening hits run model to predict 10 minutes and every minute after

# 2 cron jobs that calls the Stock Model


class Stock_Model(object):
    def __init__(self, **kwrargs):
        self.stock_symbol = kwrargs.get('stock_symbol')
        self.interval_seconds = kwrargs.get('interval_seconds')
        self.num_days = kwrargs.get('num_days')

    def load_data(self, stock, seq_len):
        amount_of_features = len(stock.columns)
        data = stock.as_matrix()  # pd.DataFrame(stock)
        sequence_length = seq_len + 1
        result = []
        for index in range(len(data) - sequence_length):
            result.append(data[index: index + sequence_length])

        result = np.array(result)
        print(result.shape)
        row = round(0.9 * result.shape[0])
        train = result[:int(row), :]

        x_train = train[:, :-1]
        y_train = train[:, -1][:, -1]
        x_test = result[int(row):, :-1]
        y_test = result[int(row):, -1][:, -1]

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))

        return [x_train, y_train, x_test, y_test]

    def run(self):
        today = model_libs.tz2ntz(datetime.utcnow(), 'UTC', 'US/Pacific').strftime('%m_%d_%y')
        time = str(datetime.now().strftime("%H_%M_%S"))
        print(time)
        q = GoogleIntradayQuote(self.stock_symbol, self.interval_seconds, self.num_days)
        q.write_csv('csv/{}/gpro_{}.csv'.format(today, time))

        data = pd.read_csv('csv/{}/gpro_{}.csv'.format(today, time), names=['Stock', 'Date', 'Time', 'Open', 'High', 'Low',
                                                                    'Current_Close', 'Volume'])
        del data['Stock']
        # Shifts the close so that we can predict the next days target value 'Future Close'
        data['Future_Close'] = data['Current_Close'].shift(-1)
        prediction_features = data.tail(1)
        # Remove the last row since we don't have a Target Value
        data = data[:-1]

        min_data = data.min()
        max_data = data.max()
        data_adj = data
        data_adj['Normalized High'] = data["High"] / (max_data["High"] - min_data["High"])
        data_adj['Normalized Low'] = data["Low"] / (max_data["Low"] - min_data["Low"])
        data_adj['Normalized Open'] = data["Open"] / (max_data["Open"] - min_data["Open"])
        data_adj['Normalized Volume'] = data["Volume"] / (max_data["Volume"] - min_data["Volume"])
        data_adj['Normalized Current_Close'] = data["Current_Close"] / (
        max_data["Current_Close"] - min_data["Current_Close"])
        # data_adj['Normalized Future_Close'] = data["Future_Close"]/(max_data["Future_Close"] - min_data["Future_Close"])
        data_adj = data_adj.drop(['Date', 'Time', 'Open', 'High', 'Low', 'Current_Close', 'Volume'], axis=1)

        window = 22
        print(data_adj.shape)
        X_train, y_train, X_test, y_test = self.load_data(data_adj[::-1], window)
        print("X_train", X_train.shape)
        print("y_train", y_train.shape)
        print("X_test", X_test.shape)
        print("y_test", y_test.shape)

        from keras.layers.core import Dense, Activation, Dropout
        from keras.layers.recurrent import LSTM
        from keras.models import Sequential
        import lstm, time  # helper libraries

        # Step 2 Build Model
        model = Sequential()
        print(X_train.shape[-1])

        model.add(LSTM(
            input_dim=X_train.shape[-1],
            output_dim=50,
            return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(
            100,
            return_sequences=False))
        model.add(Dropout(0.2))

        model.add(Dense(
            output_dim=1))
        model.add(Activation('linear'))

        start = time.time()
        model.compile(loss='mse', optimizer='rmsprop')
        print('compilation time : {}'.format(time.time() - start))

        # Step 3 Train the model
        model.fit(
            X_train,
            y_train,
            batch_size=100,
            nb_epoch=250,
            validation_split=0.05)

        trainScore = model.evaluate(X_train, y_train, verbose=0)
        print(trainScore)
        print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))

        testScore = model.evaluate(X_test, y_test, verbose=0)
        print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))


params = dict(
    stock_symbol='gpro',
    interval_seconds=600,
    num_days=30
)

a = Stock_Model(**params)

a.run()