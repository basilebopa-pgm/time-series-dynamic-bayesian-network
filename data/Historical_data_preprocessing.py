import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler


SEQUENCE_LENGTH = 10


def load_real_data(path):

    data = pd.read_csv(path)
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)

    return data


def create_features(data):

    data = data.copy()

    data["Log_return"] = np.log(data["Close"] / data["Close"].shift(1))
    data["Volume"] = np.log1p(data["Volume"])
    data["Volatility"] = (data["Log_return"].rolling(window=20).std())

    ma_20 = (data["Close"].rolling(window=20).mean())
    trend_raw = (data["Close"] / ma_20) - 1.0

    data["Trend"] = ( 1.0 / (1.0 + np.exp(-trend_raw * 100)))

    day_of_year = (data["Date"].dt.dayofyear)

    data["Seasonality"] = np.sin(2 * np.pi * day_of_year / 365.25)

    data = data[["Date", "Price", "Volume", "Volatility", "Trend", "Seasonality"]]
    data = data.dropna().reset_index(drop=True)

    return data


def split_data(data, train_ratio=0.70, validation_ratio=0.15):

    n = len(data)

    train_end = int( n * train_ratio)

    validation_end = int( n * (train_ratio + validation_ratio))

    train = data.iloc[:train_end ].copy()

    validation = data.iloc[train_end:validation_end].copy()

    test = data.iloc[ validation_end:].copy()

    return train, validation, test


def scale_data(train, validation, test):

    scaler = StandardScaler()

    columns = train.columns.drop("Date")

    train[columns] = scaler.fit_transform(train[columns])

    validation[columns] = scaler.transform( validation[columns])

    test[columns] = scaler.transform( test[columns])

    return train, validation, test, scaler


def create_dbn_sequences(data, sequence_length=SEQUENCE_LENGTH):

    data = data.drop( columns=["Date"] ).reset_index(drop=True)

    n_sequences = ( len(data) // sequence_length)

    sequences = []

    for sequence in range(n_sequences):

        start = sequence * sequence_length
        end = start + sequence_length

        block = data.iloc[ start:end]

        sequence_data = {}

        for time_slice in range(sequence_length):
            row = block.iloc[time_slice]

            for variable in data.columns:
                sequence_data[(variable, time_slice)] = row[variable]

        sequences.append(sequence_data)

    return pd.DataFrame(sequences)

def save_scaler( scaler, path="data/scaler.pkl"):

    with open(path,"wb") as file:
        pickle.dump( scaler, file)


def prepare_real_data(input_path):

    data = load_real_data(input_path)

    print("Raw observations:", len(data))

    data = create_features( data)

    print("Observations after feature engineering:", len(data))

    train, validation, test = split_data(data)

    train, validation, test, scaler = scale_data( train, validation, test)

    train_dbn = create_dbn_sequences(train)
    validation_dbn = create_dbn_sequences(validation)
    test_dbn = create_dbn_sequences(test)

    data.to_csv("data/Historical/financial_formatted.csv",index=False)

    train.to_csv("data/Historical/financial_train.csv",index=False)
    validation.to_csv("data/Historical/financial_validation.csv",index=False)
    test.to_csv("data/Historical/financial_test.csv",index=False)

    train_dbn.to_pickle("data/Historical/financial_train.pkl")
    validation_dbn.to_pickle("data/Historical/financial_validation.pkl")
    test_dbn.to_pickle("data/Historical/financial_test.pkl")

    save_scaler(scaler)

    
if __name__ == "__main__":

    prepare_real_data( "data/Historical/Financial_data.csv")