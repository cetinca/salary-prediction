import os
from itertools import combinations

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.model_selection import train_test_split

# checking ../Data directory presence
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# download data if it is unavailable
if 'data.csv' not in os.listdir('../Data'):
    url = "https://www.dropbox.com/s/3cml50uv7zm46ly/data.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/data.csv', 'wb').write(r.content)

# read data
data_ = pd.read_csv('../Data/data.csv')


# write your code here
def stage1(data):
    # alternative split method
    # data_train = data.sample(frac=0.7)
    # data_test = data.drop(data_train.index)

    # converting series to dataframe to make it 2D
    # X = pd.DataFrame(data["rating"])

    X = data[["rating"]]
    y = data["salary"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

    # model.fit does not work with pandas series, converting it to 2D array.
    # X_train = X_train.to_numpy().reshape(-1, 1)
    # X_test = X_test.to_numpy().reshape(-1, 1)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions_test = model.predict(X_test)
    predictions_mape = mape(y_test, predictions_test)

    print(model.intercept_, model.coef_[0], predictions_mape)


def stage2(data):
    # predictors 2, 3, 4
    mapes = []
    for i in [2, 3, 4]:
        # using a predictor to make the model better by raising the feature
        data["predictor"] = data["rating"] ** i

        X = data[["predictor"]]
        y = data["salary"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions_test = model.predict(X_test)
        predictions_mape = mape(y_test, predictions_test)
        mapes.append(round(predictions_mape, 5))

    # printing the best mape which is the smallest difference
    print(min(mapes))


def stage3(data):
    X = data.select_dtypes('number').drop(columns='salary')
    y = data['salary']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

    model = LinearRegression()
    model.fit(X_train, y_train)
    result = model.coef_
    print(*result, sep=', ')


def stage4(data):
    corr = data.corr(numeric_only=True)
    # print("Correlation matrix:", corr, sep="\n")
    # finding variables with high correlation with target
    variables = set()
    for index in corr.index:
        if corr.at[index, "salary"] > abs(0.2) and index != "salary":
            variables.add(index)

    X = data.drop(columns="salary")
    y = data["salary"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

    mapes = {}
    # finding model with lowest mape
    for var in variables:
        model = LinearRegression()
        model.fit(X_train.drop(columns=var), y_train)
        y_predict = model.predict(X_test.drop(columns=var))
        mape_ = mape(y_test, y_predict)
        mapes[f"{var}_dropped"] = mape_

    combins = combinations(list(variables), 2)

    for a, b in combins:
        model = LinearRegression()
        model.fit(X_train.drop(columns=[a, b]), y_train)
        y_predict = model.predict(X_test.drop(columns=[a, b]))
        mape_ = mape(y_test, y_predict)
        mapes[f"{a}_{b}_dropped"] = mape_

    mape_min_val = min(mapes.values())
    mape_min_key = [(k, v) for k, v in mapes.items() if v == min(mapes.values())]
    # print(mapes)
    # print(mape_min_key)
    # columns to keep ["draft_round", "rating", "bmi"]
    print(set(X_train.columns) - {"experience", "age"})
    print(mape_min_val)


def stage5(data):
    # dropping "experience" and "age" via last stage
    columns = ["bmi", "rating", "draft_round"]

    X = data[columns]
    # print(X.columns)
    y = data["salary"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    # print(predictions)
    # replace negative predictions with zero

    predict_1 = [0 if val < 0 else val for val in predictions]
    predict_2 = [np.median(y_test) if val < 0 else val for val in predictions]

    mape1 = mape(y_test, predict_1)
    mape2 = mape(y_test, predict_2)
    print(round(min(mape1, mape2), 5))


stage5(data_)
