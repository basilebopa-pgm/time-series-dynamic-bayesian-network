import numpy as np
import pandas as pd
import tensorflow as tf

from models.lstm import create_lstm


FEATURES = [
    "Price",
    "Volume",
    "Volatility",
    "Trend",
    "Seasonality"
]

SEQUENCE_LENGTH = 10


def load_data(path):

    data = pd.read_csv(path)

    return data[FEATURES].values


def create_sequences(data):

    X = []
    y = []

    for i in range(
        len(data) - SEQUENCE_LENGTH
    ):

        X.append(
            data[
                i:i + SEQUENCE_LENGTH
            ]
        )

        y.append(
            data[
                i + SEQUENCE_LENGTH
            ]
        )

    return (
        np.array(X),
        np.array(y)
    )


def calculate_metrics(
    y_true,
    y_pred
):

    mse = np.mean(
        (y_true - y_pred) ** 2
    )

    mae = np.mean(
        np.abs(y_true - y_pred)
    )

    rmse = np.sqrt(mse)

    ss_res = np.sum(
        (y_true - y_pred) ** 2
    )

    ss_tot = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    r2 = 1 - (
        ss_res / ss_tot
    )

    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


def calculate_per_variable_metrics(
    y_true,
    y_pred
):

    results = []

    for i, variable in enumerate(FEATURES):

        metrics = calculate_metrics(
            y_true[:, i],
            y_pred[:, i]
        )

        results.append({
            "Variable": variable,
            **metrics
        })

    return pd.DataFrame(
        results
    )


def evaluate_dataset(
    model,
    data,
    name
):

    X, y_true = create_sequences(
        data
    )

    y_pred = model.predict(
        X,
        verbose=0
    )

    overall_metrics = calculate_metrics(
        y_true,
        y_pred
    )

    variable_metrics = (
        calculate_per_variable_metrics(
            y_true,
            y_pred
        )
    )

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print()
    print("Overall metrics:")

    for metric, value in overall_metrics.items():

        print(
            f"{metric}: {value:.6f}"
        )

    print()
    print("Per-variable metrics:")

    print(
        variable_metrics.to_string(
            index=False
        )
    )

    return {
        "overall": overall_metrics,
        "per_variable": variable_metrics,
        "y_true": y_true,
        "y_pred": y_pred
    }


def evaluate_lstm():

    model = tf.keras.models.load_model(
        "models/lstm_real_trained.keras"
    )

    validation_data = load_data(
        "data/real_validation.csv"
    )

    test_data = load_data(
        "data/real_test.csv"
    )

    validation_results = evaluate_dataset(
        model,
        validation_data,
        "VALIDATION RESULTS"
    )

    test_results = evaluate_dataset(
        model,
        test_data,
        "TEST RESULTS"
    )

    return (
        validation_results,
        test_results
    )


if __name__ == "__main__":

    evaluate_lstm()