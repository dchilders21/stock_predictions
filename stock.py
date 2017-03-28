#!/bin/python python

import os
import math, time
import numpy as np
import pandas as pd
from os.path import dirname, abspath
from sklearn.preprocessing import MinMaxScaler

from quote import GoogleIntradayQuote
from datetime import datetime, timedelta
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

    def load_data(self, stock, seq_len, min_max_scaler):
        amount_of_features = len(stock.columns)
        data = stock.as_matrix()  # pd.DataFrame(stock)
        sequence_length = seq_len + 1
        result = []

        mms = min_max_scaler

        for index in range(len(data) - sequence_length):
            result.append(data[index:index + sequence_length])

        result = self.normalise_windows(result, mms)

        result = np.array(result)

        # 90% of Total
        row = round(0.9 * result.shape[0])

        train = result[:int(row), :]
        np.random.shuffle(train)
        x_train = train[:, :, :-1]

        y_train = train[:, -1][:, -1]
        x_test = result[int(row):, :, :-1]
        y_test = result[int(row):, -1][:, -1]

        # Last window we need to predict for the next interval
        future_pred = result[result.shape[0] - 1:, :, :-1]
        print(future_pred)

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features - 1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features - 1))

        return [x_train, y_train, x_test, y_test, future_pred]

    def normalise_windows(self, window_data, min_max_scaler):
        normalised_data = []

        for window in window_data:
            normalised_window = min_max_scaler.fit_transform(window)
            normalised_data.append(normalised_window)
        return normalised_data

    def run(self):
        today = model_libs.tz2ntz(datetime.utcnow(), 'UTC', 'US/Pacific').strftime('%m_%d_%y')
        time = str(datetime.now().strftime("%H_%M_%S"))

        current_path = os.path.dirname(os.path.realpath(__file__))

        if not os.path.isdir(current_path + "/csv/{}/{}".format(self.stock_symbol, today)):
            print('Making New Directory {} for the CSV'.format(today))
            os.makedirs(current_path + '/csv/{}/{}'.format(self.stock_symbol, today))

        q = GoogleIntradayQuote(self.stock_symbol, self.interval_seconds, self.num_days)

        q.write_csv(current_path + '/csv/{}/{}/{}.csv'.format(self.stock_symbol, today, time))

        data = pd.read_csv(current_path + '/csv/{}/{}/{}.csv'.format(self.stock_symbol, today, time), names=['Stock', 'Date', 'Time', 'Open', 'High', 'Low',
                                                                    'Current_Close', 'Volume'])

        del data['Stock']
        prediction_data = data.tail(1)
        # Shifts the close so that we can predict the next days target value 'Future Close'
        data['Future_Close'] = data['Current_Close'].shift(-1)

        data = data.drop(['Date', 'Time'],
                                 axis=1)

        prediction_features = data.tail(1)

        # Remove the last row since we don't have a Target Value
        data_adj = data[:-1]
        data_adj.tail()

        window = 22
        min_max_scaler = MinMaxScaler()
        X_train, y_train, X_test, y_test, future_pred = self.load_data(data_adj[::-1], window, min_max_scaler)
        print("X_train", X_train.shape)
        print("y_train", y_train.shape)
        print("X_test", X_test.shape)
        print("y_test", y_test.shape)
        print("future_pred", future_pred.shape)

        from keras.layers.core import Dense, Activation, Dropout
        from keras.layers.recurrent import LSTM
        from keras.models import Sequential
        import lstm, time  # helper libraries

        # Step 2 Build Model
        model = Sequential()

        model.add(LSTM(
            input_dim=X_train.shape[-1],
            output_dim=50,
            return_sequences=True))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(LSTM(
            100,
            return_sequences=False))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(
            output_dim=1))
        model.add(Activation('linear'))

        start = time.time()
        model.compile(loss='mse', optimizer='adam')
        print('compilation time : {}'.format(time.time() - start))

        # Step 3 Train the model
        model.fit(
            X_train,
            y_train,
            batch_size=100,
            nb_epoch=200,
            validation_split=0.05)

        trainScore = model.evaluate(X_train, y_train, verbose=0)
        print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))

        testScore = model.evaluate(X_test, y_test, verbose=0)
        print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))

        f = model.predict(future_pred)

        denormalize_prediction = min_max_scaler.scale_[5]*f[0]
        print(denormalize_prediction)
        print(' =========== ')
        prediction_data['Predicted_Future_Close'] = denormalize_prediction[0]
        print(denormalize_prediction[0])
        prediction_data['Current_Date_Time'] = prediction_data['Date'] + ' ' + prediction_data['Time']
        cdt = datetime.strptime(prediction_data.iloc[0]['Current_Date_Time'], '%Y-%m-%d %H:%M:%S')
        prediction_data["Prediction_Future_Date"] = cdt + timedelta(minutes=10)

        prediction_data = prediction_data.drop(['Date', 'Time', 'Open', 'High', 'Low', 'Volume'], axis=1)
        prediction_data = prediction_data[['Current_Date_Time', 'Current_Close', 'Prediction_Future_Date', 'Predicted_Future_Close']]

        if not os.path.isfile(current_path + "/csv/{}/{}/prediction.csv".format(self.stock_symbol, today)):
            prediction_data.to_csv(current_path + "/csv/{}/{}/prediction.csv".format(self.stock_symbol, today))
        else:
            print('Adding to Existing CSV')
            data = pd.read_csv(current_path + "/csv/{}/{}/prediction.csv".format(self.stock_symbol, today))
            data = data[['Current_Date_Time', 'Current_Close', 'Prediction_Future_Date', 'Predicted_Future_Close']]

            new_data = pd.concat([data, prediction_data])
            new_data.to_csv(current_path + "/csv/{}/{}/prediction.csv".format(self.stock_symbol, today))



params = dict(
    stock_symbol='gpro',
    interval_seconds=600,
    num_days=30
)

a = Stock_Model(**params)

a.run()