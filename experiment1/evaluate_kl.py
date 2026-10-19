import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from models.DLGBN import DynamicNode
from models.ground_truth_model import create_ground_truth


N_TIME_SLICES = 10

VARIABLES = [
    "Price",
    "Volume",
    "Volatility",
    "Trend"
]


def gaussian_kl(
    mu_true,
    std_true,
    mu_model,
    std_model
):

    return (
        np.log(std_model / std_true)
        + (
            std_true**2
            + (mu_true - mu_model)**2
        ) / (2 * std_model**2)
        - 0.5
    )


def conditional_mean(cpd, row, t):

    mean = float(cpd.beta[0])

    for beta, parent in zip(
        cpd.beta[1:],
        cpd.evidence
    ):

        if parent[1] == 0:

            value = row[
                (parent[0], t - 1)
            ]

        else:

            value = row[
                (parent[0], t)
            ]

        mean += (
            float(beta)
            * float(value)
        )

    return mean


def calculate_dbn_kl(
    true_model,
    fitted_model,
    test_data
):

    results = {
        variable: []
        for variable in VARIABLES
    }

    for _, row in test_data.iterrows():

        for t in range(
            1,
            N_TIME_SLICES
        ):

            for variable in VARIABLES:

                true_cpd = true_model.get_cpds(
                    node=DynamicNode(
                        variable,
                        1
                    )
                )

                model_cpd = fitted_model.get_cpds(
                    node=DynamicNode(
                        variable,
                        1
                    )
                )

                mu_true = conditional_mean(
                    true_cpd,
                    row,
                    t
                )

                mu_model = conditional_mean(
                    model_cpd,
                    row,
                    t
                )

                kl = gaussian_kl(
                    mu_true,
                    true_cpd.std,
                    mu_model,
                    model_cpd.std
                )

                results[variable].append(
                    {
                        "time": t,
                        "kl": kl
                    }
                )

    return results


def prepare_lstm_data(data):

    X = []
    y = []

    for _, row in data.iterrows():

        sequence = []

        for t in range(
            N_TIME_SLICES
        ):

            values = [
                row[
                    (variable, t)
                ]
                for variable in VARIABLES
            ]

            sequence.append(values)

        sequence = np.array(sequence)

        X.append(
            sequence[:-1]
        )

        y.append(
            sequence[1:]
        )

    return (
        np.array(X),
        np.array(y)
    )


def calculate_lstm_kl(
    true_model,
    lstm,
    test_data
):

    X_test, y_test = prepare_lstm_data(
        test_data
    )

    predictions = lstm.predict(
        X_test,
        verbose=0
    )

    results = {
        variable: []
        for variable in VARIABLES
    }

    for variable_index, variable in enumerate(
        VARIABLES
    ):

        residuals = (
            y_test[:, :, variable_index]
            - predictions[:, :, variable_index]
        )

        model_std = np.std(
            residuals,
            ddof=1
        )

        true_cpd = true_model.get_cpds(
            node=DynamicNode(
                variable,
                1
            )
        )

        for sequence_index in range(
            len(test_data)
        ):

            row = test_data.iloc[
                sequence_index
            ]

            for t in range(
                1,
                N_TIME_SLICES
            ):

                mu_true = conditional_mean(
                    true_cpd,
                    row,
                    t
                )

                mu_model = predictions[
                    sequence_index,
                    t - 1,
                    variable_index
                ]

                kl = gaussian_kl(
                    mu_true,
                    true_cpd.std,
                    mu_model,
                    model_std
                )

                results[variable].append(
                    {
                        "time": t,
                        "kl": kl
                    }
                )

    return results


def average_kl(results):

    return {
        variable: np.mean(
            [
                item["kl"]
                for item in results[variable]
            ]
        )
        for variable in VARIABLES
    }


def plot_average_kl(
    model1_avg,
    model2_avg,
    lstm_avg
):

    x = np.arange(
        len(VARIABLES)
    )

    width = 0.25

    model1_values = [
        model1_avg[variable]
        for variable in VARIABLES
    ]

    model2_values = [
        model2_avg[variable]
        for variable in VARIABLES
    ]

    lstm_values = [
        lstm_avg[variable]
        for variable in VARIABLES
    ]

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        x - width,
        model1_values,
        width,
        label="Model 1"
    )

    plt.bar(
        x,
        model2_values,
        width,
        label="Model 2"
    )

    plt.bar(
        x + width,
        lstm_values,
        width,
        label="LSTM"
    )

    plt.xlabel(
        "Variable"
    )

    plt.ylabel(
        "Average KL Divergence"
    )

    plt.title(
        "Average KL Divergence by Variable"
    )

    plt.xticks(
        x,
        VARIABLES
    )

    plt.yscale(
        "log"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "experiments/kl_three_models.png",
        dpi=300
    )

    plt.show()


def plot_kl_over_time(
    model1_results,
    model2_results,
    lstm_results
):

    for variable in VARIABLES:

        model1_by_time = {}
        model2_by_time = {}
        lstm_by_time = {}

        for item in model1_results[variable]:

            t = item["time"]

            if t not in model1_by_time:

                model1_by_time[t] = []

            model1_by_time[t].append(
                item["kl"]
            )

        for item in model2_results[variable]:

            t = item["time"]

            if t not in model2_by_time:

                model2_by_time[t] = []

            model2_by_time[t].append(
                item["kl"]
            )

        for item in lstm_results[variable]:

            t = item["time"]

            if t not in lstm_by_time:

                lstm_by_time[t] = []

            lstm_by_time[t].append(
                item["kl"]
            )

        times = sorted(
            model1_by_time.keys()
        )

        model1_values = [
            np.mean(
                model1_by_time[t]
            )
            for t in times
        ]

        model2_values = [
            np.mean(
                model2_by_time[t]
            )
            for t in times
        ]

        lstm_values = [
            np.mean(
                lstm_by_time[t]
            )
            for t in times
        ]

        plt.figure(
            figsize=(10, 6)
        )

        plt.plot(
            times,
            model1_values,
            marker="o",
            label="Model 1"
        )

        plt.plot(
            times,
            model2_values,
            marker="o",
            label="Model 2"
        )

        plt.plot(
            times,
            lstm_values,
            marker="o",
            label="LSTM"
        )

        plt.xlabel(
            "Time Slice"
        )

        plt.ylabel(
            "Average KL Divergence"
        )

        plt.title(
            f"KL Divergence Over Time - {variable}"
        )

        plt.xticks(
            times
        )

        plt.yscale(
            "log"
        )

        plt.legend()

        plt.tight_layout()

        filename = (
            "experiments/"
            f"kl_over_time_"
            f"{variable.lower()}_three_models.png"
        )

        plt.savefig(
            filename,
            dpi=300
        )

        plt.show()


if __name__ == "__main__":

    print("Loading test data...")

    test_data = pd.read_pickle(
        "data/Simulation/GT_test.pkl"
    )

    print("Creating ground truth...")

    ground_truth = create_ground_truth()

    print("Loading Model 1...")

    with open(
        "models/Simulation/model1_1st_order_trained.pkl",
        "rb"
    ) as f:

        model_1 = pickle.load(f)

    print("Loading Model 2...")

    with open(
        "models/Simulation/model2_1st_order_seasonal_trained.pkl",
        "rb"
    ) as f:

        model_2 = pickle.load(f)

    print("Loading LSTM...")

    lstm = load_model(
        "models/lstm_trained.keras"
    )

    print("\nCalculating Model 1 KL...")

    model1_results = calculate_dbn_kl(
        ground_truth,
        model_1,
        test_data
    )

    print("Calculating Model 2 KL...")

    model2_results = calculate_dbn_kl(
        ground_truth,
        model_2,
        test_data
    )

    print("Calculating LSTM KL...")

    lstm_results = calculate_lstm_kl(
        ground_truth,
        lstm,
        test_data
    )

    model1_avg = average_kl(
        model1_results
    )

    model2_avg = average_kl(
        model2_results
    )

    lstm_avg = average_kl(
        lstm_results
    )

    print("\nKL Divergence Results")
    print("-" * 65)

    print(
        f"{'Variable':<15}"
        f"{'Model 1':>15}"
        f"{'Model 2':>15}"
        f"{'LSTM':>15}"
    )

    print("-" * 65)

    for variable in VARIABLES:

        print(
            f"{variable:<15}"
            f"{model1_avg[variable]:>15.6f}"
            f"{model2_avg[variable]:>15.6f}"
            f"{lstm_avg[variable]:>15.6f}"
        )

    print("\nOverall KL")
    print("-" * 30)

    print(
        f"Model 1: "
        f"{np.mean(list(model1_avg.values())):.6f}"
    )

    print(
        f"Model 2: "
        f"{np.mean(list(model2_avg.values())):.6f}"
    )

    print(
        f"LSTM: "
        f"{np.mean(list(lstm_avg.values())):.6f}"
    )

    print("\nGenerating average KL plot...")

    plot_average_kl(
        model1_avg,
        model2_avg,
        lstm_avg
    )

    print("\nGenerating KL-over-time plots...")

    plot_kl_over_time(
        model1_results,
        model2_results,
        lstm_results
    )

    print("\nEvaluation complete.")