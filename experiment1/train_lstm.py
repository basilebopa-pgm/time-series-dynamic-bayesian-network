import numpy as np
import pandas as pd

from models.lstm import create_lstm


VARIABLES = [
    "Price",
    "Volume",
    "Volatility",
    "Trend"
]

N_TIME_SLICES = 10


def load_training_data():

    return pd.read_pickle(
        "data/train.pkl"
    )


def prepare_sequences(data):

    X = []
    y = []

    for _, row in data.iterrows():

        sequence = []

        for t in range(N_TIME_SLICES):

            values = [
                row[(variable, t)]
                for variable in VARIABLES
            ]

            sequence.append(values)

        sequence = np.array(sequence)

        X.append(sequence[:-1])
        y.append(sequence[1:])

    return np.array(X), np.array(y)


def train_model():

    data = load_training_data()

    X_train, y_train = prepare_sequences(
        data
    )

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    model = create_lstm(
        input_shape=(
            N_TIME_SLICES - 1,
            len(VARIABLES)
        ),
        output_size=len(VARIABLES)
    )

    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=32,
        shuffle=False
    )

    model.save(
        "models/lstm_trained.keras"
    )

    return model


if __name__ == "__main__":

    model = train_model()

    print("LSTM trained successfully.")