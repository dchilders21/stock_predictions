from flask import Flask, render_template, request, url_for, redirect, jsonify
import pandas as pd
import model_libs
from datetime import datetime, timedelta
app = Flask(__name__)

@app.route("/")
def main():
    today = model_libs.tz2ntz(datetime.utcnow(), 'UTC', 'US/Pacific').strftime('%m_%d_%y')
    data = pd.read_csv('csv/gpro/{}/prediction.csv'.format(today))

    info = data[['Current_Date_Time', 'Current_Close']].as_matrix()
    info = info.tolist()

    predicted = data[['Prediction_Future_Date', 'Predicted_Future_Close']].as_matrix()
    predicted = predicted.tolist()

    maxPrice = max(info + predicted, key=lambda x: x[1])
    minPrice = min(info + predicted, key=lambda x: x[1])

    return render_template('stocks.html', info=info, predicted=predicted, minPrice=minPrice[1], maxPrice=maxPrice[1])

if __name__ == "__main__":
    app.run()