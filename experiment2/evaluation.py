import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

from models.DLGBN import DynamicNode


MODEL1_FEATURES = [
    "Price",
    "Volume",
    "Volatility",
    "Trend"
]

MODEL2_FEATURES = [
    "Price",
    "Volume",
    "Volatility",
    "Trend",
    "Seasonality"
]

SEQUENCE_LENGTH = 10


def load_model1():

    with open(
        "models/model1_real_trained.pkl",
        "rb"
    ) as file:

        return pickle.load(file)


def load_model2():

    with open(
        "models/model2_real_trained.pkl",
        "rb"
    ) as file:

        return pickle.load(file)


def load_lstm():

    return tf.keras.models.load_model(
        "models/lstm_real_trained.keras"
    )


def load_data(split):

    return pd.read_csv(
        f"data/real_{split}.csv"
    )


def get_cpd(
    model,
    variable,
    time_slice
):

    node = DynamicNode(
        variable,
        time_slice
    )

    cpd = model.get_cpds(
        node=node
    )

    if cpd is None:

        raise ValueError(
            f"CPD not found for "
            f"{variable}, time slice {time_slice}"
        )

    return cpd


def get_evidence_value(
    evidence,
    values
):

    return values[
        (
            evidence.variable,
            evidence.time_slice
        )
    ]


def get_cpd_prediction(
    cpd,
    values
):

    mean = float(
        cpd.beta[0]
    )

    for i, evidence in enumerate(
        cpd.evidence
    ):

        value = get_evidence_value(
            evidence,
            values
        )

        mean += (
            float(cpd.beta[i + 1])
            * value
        )

    return (
        mean,
        float(cpd.std)
    )


def predict_dbn_sequence(
    model,
    sequence,
    features
):

    predictions = []
    actuals = []
    stds = []

    for t in range(
        1,
        len(sequence)
    ):

        values = {}

        for time_slice in range(t):

            row = sequence[
                time_slice
            ]

            for feature in features:

                values[
                    (
                        feature,
                        time_slice
                    )
                ] = row[feature]

        predicted_values = []
        predicted_stds = []
        actual_values = []

        for feature in features:

            cpd = get_cpd(
                model,
                feature,
                1
            )

            mean, std = (
                get_cpd_prediction(
                    cpd,
                    values
                )
            )

            predicted_values.append(
                mean
            )

            predicted_stds.append(
                std
            )

            actual_values.append(
                sequence[t][feature]
            )

        predictions.append(
            predicted_values
        )

        stds.append(
            predicted_stds
        )

        actuals.append(
            actual_values
        )

    return (
        np.asarray(actuals),
        np.asarray(predictions),
        np.asarray(stds)
    )


def evaluate_dbn(
    model,
    data,
    features,
    model_name
):

    values = data[
        features
    ].to_dict(
        orient="records"
    )

    all_actual = []
    all_predictions = []
    all_stds = []

    n_sequences = (
        len(values)
        // SEQUENCE_LENGTH
    )

    for sequence_index in range(
        n_sequences
    ):

        start = (
            sequence_index
            * SEQUENCE_LENGTH
        )

        end = (
            start
            + SEQUENCE_LENGTH
        )

        sequence = values[
            start:end
        ]

        actual, predictions, stds = (
            predict_dbn_sequence(
                model,
                sequence,
                features
            )
        )

        all_actual.append(
            actual
        )

        all_predictions.append(
            predictions
        )

        all_stds.append(
            stds
        )

    if not all_actual:

        raise ValueError(
            "No complete DBN sequences "
            "were found."
        )

    y_true = np.vstack(
        all_actual
    )

    y_pred = np.vstack(
        all_predictions
    )

    y_std = np.vstack(
        all_stds
    )

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    variable_metrics = (
        calculate_variable_metrics(
            y_true,
            y_pred,
            features
        )
    )

    nll_values = []

    for i in range(
        len(features)
    ):

        nll = gaussian_nll(
            y_true[:, i],
            y_pred[:, i],
            y_std[:, i]
        )

        nll_values.append(
            nll
        )

    variable_metrics["NLL"] = (
        nll_values
    )

    overall_nll = np.mean(
        nll_values
    )

    print()
    print("=" * 70)
    print(model_name)
    print("=" * 70)

    print()
    print("Overall metrics:")

    print(
        f"MSE:  {metrics['MSE']:.6f}"
    )

    print(
        f"MAE:  {metrics['MAE']:.6f}"
    )

    print(
        f"RMSE: {metrics['RMSE']:.6f}"
    )

    print(
        f"R2:   {metrics['R2']:.6f}"
    )

    print(
        f"NLL:  {overall_nll:.6f}"
    )

    print()
    print("Per-variable metrics:")

    print(
        variable_metrics.to_string(
            index=False
        )
    )

    return {
        "overall": metrics,
        "NLL": overall_nll,
        "per_variable": variable_metrics
    }


def create_lstm_sequences(
    data,
    features
):

    values = data[
        features
    ].values

    X = []
    y = []

    for i in range(
        len(values)
        - SEQUENCE_LENGTH
    ):

        X.append(
            values[
                i:i + SEQUENCE_LENGTH
            ]
        )

        y.append(
            values[
                i + SEQUENCE_LENGTH
            ]
        )

    return (
        np.asarray(X),
        np.asarray(y)
    )


def evaluate_lstm(
    model,
    data,
    features,
    model_name
):

    X, y_true = (
        create_lstm_sequences(
            data,
            features
        )
    )

    y_pred = model.predict(
        X,
        verbose=0
    )

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    variable_metrics = (
        calculate_variable_metrics(
            y_true,
            y_pred,
            features
        )
    )

    print()
    print("=" * 70)
    print(model_name)
    print("=" * 70)

    print()
    print("Overall metrics:")

    print(
        f"MSE:  {metrics['MSE']:.6f}"
    )

    print(
        f"MAE:  {metrics['MAE']:.6f}"
    )

    print(
        f"RMSE: {metrics['RMSE']:.6f}"
    )

    print(
        f"R2:   {metrics['R2']:.6f}"
    )

    print()
    print("Per-variable metrics:")

    print(
        variable_metrics.to_string(
            index=False
        )
    )

    return {
        "overall": metrics,
        "per_variable": variable_metrics
    }


def calculate_metrics(
    y_true,
    y_pred
):

    mse = np.mean(
        (
            y_true
            - y_pred
        ) ** 2
    )

    mae = np.mean(
        np.abs(
            y_true
            - y_pred
        )
    )

    rmse = np.sqrt(
        mse
    )

    ss_res = np.sum(
        (
            y_true
            - y_pred
        ) ** 2
    )

    ss_tot = np.sum(
        (
            y_true
            - np.mean(
                y_true,
                axis=0
            )
        ) ** 2
    )

    if ss_tot == 0:

        r2 = np.nan

    else:

        r2 = (
            1
            - ss_res / ss_tot
        )

    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


def calculate_variable_metrics(
    y_true,
    y_pred,
    features
):

    results = []

    for i, feature in enumerate(
        features
    ):

        metrics = calculate_metrics(
            y_true[:, i],
            y_pred[:, i]
        )

        results.append({
            "Variable": feature,
            "MSE": metrics["MSE"],
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "R2": metrics["R2"]
        })

    return pd.DataFrame(
        results
    )


def gaussian_nll(
    y_true,
    mean,
    std
):

    std = np.maximum(
        std,
        1e-8
    )

    nll = (
        0.5
        * np.log(
            2
            * np.pi
            * std ** 2
        )
        +
        (
            (
                y_true
                - mean
            ) ** 2
            /
            (
                2
                * std ** 2
            )
        )
    )

    return np.mean(
        nll
    )


def evaluate_split(
    split,
    model1,
    model2,
    lstm
):

    data = load_data(
        split
    )

    print()
    print()
    print("#" * 70)
    print(
        f"{split.upper()} DATA"
    )
    print("#" * 70)

    model1_results = evaluate_dbn(
        model1,
        data,
        MODEL1_FEATURES,
        "MODEL 1"
    )

    model2_results = evaluate_dbn(
        model2,
        data,
        MODEL2_FEATURES,
        "MODEL 2"
    )

    lstm_results = evaluate_lstm(
        lstm,
        data,
        MODEL2_FEATURES,
        "LSTM"
    )

    return {
        "Model 1": model1_results,
        "Model 2": model2_results,
        "LSTM": lstm_results
    }


def create_summary(
    validation_results,
    test_results
):

    rows = []

    results = [
        (
            "Validation",
            validation_results
        ),
        (
            "Test",
            test_results
        )
    ]

    for dataset_name, dataset_results in results:

        for model_name, result in (
            dataset_results.items()
        ):

            row = {
                "Dataset": dataset_name,
                "Model": model_name,
                "MSE": result[
                    "overall"
                ]["MSE"],
                "MAE": result[
                    "overall"
                ]["MAE"],
                "RMSE": result[
                    "overall"
                ]["RMSE"],
                "R2": result[
                    "overall"
                ]["R2"]
            }

            if "NLL" in result:

                row["NLL"] = result[
                    "NLL"
                ]

            else:

                row["NLL"] = np.nan

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


def main():

    print("Loading models...")

    model1 = load_model1()

    model2 = load_model2()

    lstm = load_lstm()

    validation_results = (
        evaluate_split(
            "validation",
            model1,
            model2,
            lstm
        )
    )

    test_results = (
        evaluate_split(
            "test",
            model1,
            model2,
            lstm
        )
    )

    summary = create_summary(
        validation_results,
        test_results
    )

    print()
    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    summary.to_csv(
        "data/real_model_results.csv",
        index=False
    )

    print()
    print(
        "Results saved to "
        "data/real_model_results.csv"
    )


if __name__ == "__main__":

    main()