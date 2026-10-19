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

EPOCHS = 20
BATCH_SIZE = 32

N_FOLDS = 5


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


def create_model():

    return create_lstm(
        input_shape=(
            N_TIME_SLICES - 1,
            len(VARIABLES)
        ),
        output_size=len(VARIABLES)
    )


def walk_forward_validation(X, y):

    n_samples = len(X)

    fold_size = n_samples // (N_FOLDS + 1)

    validation_losses = []

    for fold in range(N_FOLDS):

        train_end = fold_size * (fold + 1)
        validation_end = fold_size * (fold + 2)

        X_train = X[:train_end]
        y_train = y[:train_end]

        X_val = X[train_end:validation_end]
        y_val = y[train_end:validation_end]

        print()
        print("=" * 50)
        print(f"Fold {fold + 1}/{N_FOLDS}")
        print("=" * 50)

        print("Training samples:", len(X_train))
        print("Validation samples:", len(X_val))

        model = create_model()

        history = model.fit(
            X_train,
            y_train,
            validation_data=(
                X_val,
                y_val
            ),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=False,
            verbose=1
        )

        val_loss = min(
            history.history["val_loss"]
        )

        validation_losses.append(
            val_loss
        )

        print(
            f"Best validation loss: {val_loss:.6f}"
        )

    print()
    print("=" * 50)
    print("Walk-forward results")
    print("=" * 50)

    for i, loss in enumerate(
        validation_losses,
        start=1
    ):

        print(
            f"Fold {i}: {loss:.6f}"
        )

    print(
        f"\nMean validation loss: "
        f"{np.mean(validation_losses):.6f}"
    )

    return validation_losses


def train_final_model(X, y):

    print()
    print("=" * 50)
    print("Training final LSTM")
    print("=" * 50)

    model = create_model()

    model.fit(
        X,
        y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        verbose=1
    )

    model.save(
        "models/lstm_trained_walk_forward.keras"
    )

    return model


def train_model():

    data = load_training_data()

    X, y = prepare_sequences(data)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Walk-forward validation
    validation_losses = walk_forward_validation(
        X,
        y
    )

    # After validation, train final model
    # using all available training data
    model = train_final_model(
        X,
        y
    )

    return model, validation_losses


if __name__ == "__main__":

    model, validation_losses = train_model()

    print()
    print("LSTM trained successfully.")