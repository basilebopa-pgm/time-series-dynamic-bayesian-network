import numpy as np
import pandas as pd
import pickle
from scipy.stats import wasserstein_distance
from models.DLGBN import DynamicNode
from tensorflow.keras.models import load_model


# ============================================================
# SETTINGS
# ============================================================

VARIABLES = [
    "Price",
    "Volume",
    "Volatility",
    "Trend"
]

TEST_PATH = "data/test.pkl"
MODEL1_PATH = "models/model1_trained.pkl"
MODEL2_PATH = "models/model2_trained.pkl"
LSTM_PATH = "models/lstm_trained.keras"


# ============================================================
# LOAD DATA
# ============================================================

with open(TEST_PATH, "rb") as f:
    test_data = pickle.load(f)

with open(MODEL1_PATH, "rb") as f:
    model1 = pickle.load(f)

with open(MODEL2_PATH, "rb") as f:
    model2 = pickle.load(f)

lstm = load_model(LSTM_PATH)


# ============================================================
# CONVERT TEST DATA TO DATAFRAME
# ============================================================

if isinstance(test_data, pd.DataFrame):
    test_df = test_data.copy()
else:
    test_df = pd.DataFrame(test_data)


# ============================================================
# GET TIME SLICES
# ============================================================

time_slices = sorted(
    set(
        int(col[1])
        for col in test_df.columns
        if isinstance(col, tuple)
    )
)

max_time = max(time_slices)


# ============================================================
# CONDITIONAL MEAN
# ============================================================

def conditional_mean(cpd, row, t):

    mean = float(cpd.beta[0])

    for beta, parent in zip(cpd.beta[1:], cpd.evidence):

        if parent[1] == 0:
            value = row[(parent[0], t - 1)]
        else:
            value = row[(parent[0], t)]

        mean += float(beta) * float(value)

    return mean


# ============================================================
# GET TRUE CONDITIONAL MEANS
# ============================================================

def get_true_means():

    true_means = {
        variable: []
        for variable in VARIABLES
    }

    for _, row in test_df.iterrows():

        for t in range(1, max_time + 1):

            for variable in VARIABLES:

                cpd = model1.get_cpds(
                    DynamicNode(variable, t)
                )

                mu = conditional_mean(cpd, row, t)

                true_means[variable].append(mu)

    return true_means


# ============================================================
# GET DBN PREDICTIONS
# ============================================================

def get_dbn_predictions(model):

    predictions = {
        variable: []
        for variable in VARIABLES
    }

    for _, row in test_df.iterrows():

        for t in range(1, max_time + 1):

            for variable in VARIABLES:

                cpd = model.get_cpds(
                    DynamicNode(variable, t)
                )

                mu = conditional_mean(cpd, row, t)

                predictions[variable].append(mu)

    return predictions


# ============================================================
# PREPARE LSTM DATA
# ============================================================

def prepare_lstm_data(df):

    X = []
    y = []

    for _, row in df.iterrows():

        sequence = []

        for t in range(max_time + 1):

            values = []

            for variable in VARIABLES:
                values.append(row[(variable, t)])

            sequence.append(values)

        sequence = np.array(sequence, dtype=np.float32)

        X.append(sequence[:-1])
        y.append(sequence[1:])

    return np.array(X), np.array(y)


X_test, y_test = prepare_lstm_data(test_df)

lstm_predictions = lstm.predict(
    X_test,
    verbose=0
)


# ============================================================
# LSTM PREDICTIONS
# ============================================================

lstm_pred = {
    variable: []
    for variable in VARIABLES
}

for variable_index, variable in enumerate(VARIABLES):

    lstm_pred[variable] = (
        lstm_predictions[:, :, variable_index]
        .flatten()
        .tolist()
    )


# ============================================================
# TRUE VALUES FOR LSTM
# ============================================================

lstm_true = {
    variable: []
    for variable in VARIABLES
}

for variable_index, variable in enumerate(VARIABLES):

    lstm_true[variable] = (
        y_test[:, :, variable_index]
        .flatten()
        .tolist()
    )


# ============================================================
# TRUE CONDITIONAL MEANS
# ============================================================

true_means = get_true_means()


# ============================================================
# MAE
# ============================================================

def calculate_mae(true_values, predictions):

    return np.mean(
        np.abs(
            np.array(true_values)
            - np.array(predictions)
        )
    )


# ============================================================
# RMSE
# ============================================================

def calculate_rmse(true_values, predictions):

    return np.sqrt(
        np.mean(
            (
                np.array(true_values)
                - np.array(predictions)
            ) ** 2
        )
    )


# ============================================================
# STANDARDISED MAE
# ============================================================

def calculate_standardised_mae(
    true_values,
    predictions,
    true_std
):

    return calculate_mae(
        true_values,
        predictions
    ) / true_std


# ============================================================
# STANDARDISED RMSE
# ============================================================

def calculate_standardised_rmse(
    true_values,
    predictions,
    true_std
):

    return calculate_rmse(
        true_values,
        predictions
    ) / true_std


# ============================================================
# WASSERSTEIN DISTANCE
# ============================================================

def calculate_wasserstein(
    true_values,
    predictions
):

    return wasserstein_distance(
        true_values,
        predictions
    )


# ============================================================
# GAUSSIAN KL
# ============================================================

def gaussian_kl(
    mu_true,
    std_true,
    mu_model,
    std_model
):

    return (
        np.log(std_model / std_true)
        + (
            std_true ** 2
            + (mu_true - mu_model) ** 2
        )
        / (2 * std_model ** 2)
        - 0.5
    )


# ============================================================
# DBN KL
# ============================================================

def calculate_dbn_kl(model):

    results = {
        variable: []
        for variable in VARIABLES
    }

    for _, row in test_df.iterrows():

        for t in range(1, max_time + 1):

            for variable in VARIABLES:

                true_cpd = model1.get_cpds(
                    DynamicNode(variable, t)
                )

                model_cpd = model.get_cpds(
                    DynamicNode(variable, t)
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

                std_true = float(true_cpd.std)
                std_model = float(model_cpd.std)

                kl = gaussian_kl(
                    mu_true,
                    std_true,
                    mu_model,
                    std_model
                )

                results[variable].append(kl)

    return {
        variable: np.mean(values)
        for variable, values in results.items()
    }


# ============================================================
# LSTM KL
# ============================================================
# IMPORTANT:
# One global residual standard deviation is calculated
# for each variable across all test sequences and time steps.
# This is the ORIGINAL method used previously.
# ============================================================

def calculate_lstm_kl():

    results = {}

    for variable_index, variable in enumerate(VARIABLES):

        # Global residuals for this variable
        residuals = (
            y_test[:, :, variable_index]
            - lstm_predictions[:, :, variable_index]
        )

        # ONE standard deviation for the entire variable
        model_std = np.std(
            residuals,
            ddof=1
        )

        kl_values = []

        for sequence_index in range(
            len(test_df)
        ):

            row = test_df.iloc[sequence_index]

            for t in range(1, max_time + 1):

                true_cpd = model1.get_cpds(
                    DynamicNode(variable, t)
                )

                mu_true = conditional_mean(
                    true_cpd,
                    row,
                    t
                )

                # LSTM prediction for t
                mu_model = lstm_predictions[
                    sequence_index,
                    t - 1,
                    variable_index
                ]

                std_true = float(
                    true_cpd.std
                )

                kl = gaussian_kl(
                    mu_true,
                    std_true,
                    mu_model,
                    model_std
                )

                kl_values.append(kl)

        results[variable] = np.mean(
            kl_values
        )

    return results


# ============================================================
# STANDARD DEVIATIONS
# ============================================================

true_stds = {}

for variable in VARIABLES:

    true_stds[variable] = np.std(
        true_means[variable],
        ddof=1
    )


# ============================================================
# GET MODEL PREDICTIONS
# ============================================================

model1_predictions = get_dbn_predictions(
    model1
)

model2_predictions = get_dbn_predictions(
    model2
)


# ============================================================
# CALCULATE MAE
# ============================================================

mae_model1 = {}
mae_model2 = {}
mae_lstm = {}

for variable in VARIABLES:

    mae_model1[variable] = calculate_mae(
        true_means[variable],
        model1_predictions[variable]
    )

    mae_model2[variable] = calculate_mae(
        true_means[variable],
        model2_predictions[variable]
    )

    mae_lstm[variable] = calculate_mae(
        true_means[variable],
        lstm_pred[variable]
    )


# ============================================================
# CALCULATE RMSE
# ============================================================

rmse_model1 = {}
rmse_model2 = {}
rmse_lstm = {}

for variable in VARIABLES:

    rmse_model1[variable] = calculate_rmse(
        true_means[variable],
        model1_predictions[variable]
    )

    rmse_model2[variable] = calculate_rmse(
        true_means[variable],
        model2_predictions[variable]
    )

    rmse_lstm[variable] = calculate_rmse(
        true_means[variable],
        lstm_pred[variable]
    )


# ============================================================
# STANDARDISED MAE
# ============================================================

smae_model1 = {}
smae_model2 = {}
smae_lstm = {}

for variable in VARIABLES:

    smae_model1[variable] = (
        mae_model1[variable]
        / true_stds[variable]
    )

    smae_model2[variable] = (
        mae_model2[variable]
        / true_stds[variable]
    )

    smae_lstm[variable] = (
        mae_lstm[variable]
        / true_stds[variable]
    )


# ============================================================
# STANDARDISED RMSE
# ============================================================

srmse_model1 = {}
srmse_model2 = {}
srmse_lstm = {}

for variable in VARIABLES:

    srmse_model1[variable] = (
        rmse_model1[variable]
        / true_stds[variable]
    )

    srmse_model2[variable] = (
        rmse_model2[variable]
        / true_stds[variable]
    )

    srmse_lstm[variable] = (
        rmse_lstm[variable]
        / true_stds[variable]
    )


# ============================================================
# WASSERSTEIN
# ============================================================

wasserstein_model1 = {}
wasserstein_model2 = {}
wasserstein_lstm = {}

for variable in VARIABLES:

    wasserstein_model1[variable] = (
        calculate_wasserstein(
            true_means[variable],
            model1_predictions[variable]
        )
    )

    wasserstein_model2[variable] = (
        calculate_wasserstein(
            true_means[variable],
            model2_predictions[variable]
        )
    )

    wasserstein_lstm[variable] = (
        calculate_wasserstein(
            true_means[variable],
            lstm_pred[variable]
        )
    )


# ============================================================
# STANDARDISED WASSERSTEIN
# ============================================================

swasserstein_model1 = {}
swasserstein_model2 = {}
swasserstein_lstm = {}

for variable in VARIABLES:

    swasserstein_model1[variable] = (
        wasserstein_model1[variable]
        / true_stds[variable]
    )

    swasserstein_model2[variable] = (
        wasserstein_model2[variable]
        / true_stds[variable]
    )

    swasserstein_lstm[variable] = (
        wasserstein_lstm[variable]
        / true_stds[variable]
    )


# ============================================================
# KL DIVERGENCE
# ============================================================

print("\nCalculating KL divergence...")

kl_model1 = calculate_dbn_kl(
    model1
)

kl_model2 = calculate_dbn_kl(
    model2
)

kl_lstm = calculate_lstm_kl()


# ============================================================
# PRINT MAE
# ============================================================

print("\nMAE")
print("-" * 85)
print(
    f"{'Variable':<20}"
    f"{'Model 1':>15}"
    f"{'Model 2':>18}"
    f"{'LSTM':>18}"
)
print("-" * 85)

for variable in VARIABLES:

    print(
        f"{variable:<20}"
        f"{mae_model1[variable]:>15.6f}"
        f"{mae_model2[variable]:>18.6f}"
        f"{mae_lstm[variable]:>18.6f}"
    )


# ============================================================
# PRINT RMSE
# ============================================================

print("\nRMSE")
print("-" * 85)
print(
    f"{'Variable':<20}"
    f"{'Model 1':>15}"
    f"{'Model 2':>18}"
    f"{'LSTM':>18}"
)
print("-" * 85)

for variable in VARIABLES:

    print(
        f"{variable:<20}"
        f"{rmse_model1[variable]:>15.6f}"
        f"{rmse_model2[variable]:>18.6f}"
        f"{rmse_lstm[variable]:>18.6f}"
    )


# ============================================================
# PRINT STANDARDISED MAE
# ============================================================

print("\nStandardised MAE")
print("-" * 85)
print(
    f"{'Variable':<20}"
    f"{'Model 1':>15}"
    f"{'Model 2':>18}"
    f"{'LSTM':>18}"
)
print("-" * 85)

for variable in VARIABLES:

    print(
        f"{variable:<20}"
        f"{smae_model1[variable]:>15.6f}"
        f"{smae_model2[variable]:>18.6f}"
        f"{smae_lstm[variable]:>18.6f}"
    )


# ============================================================
# PRINT STANDARDISED RMSE
# ============================================================

print("\nStandardised RMSE")
print("-" * 85)
print(
    f"{'Variable':<20}"
    f"{'Model 1':>15}"
    f"{'Model 2':>18}"
    f"{'LSTM':>18}"
)
print("-" * 85)

for variable in VARIABLES:

    print(
        f"{variable:<20}"
        f"{srmse_model1[variable]:>15.6f}"
        f"{srmse_model2[variable]:>18.6f}"
        f"{srmse_lstm[variable]:>18.6f}"
    )


# ============================================================
# PRINT WASSERSTEIN
# ============================================================

print("\nWasserstein Distance")
print("-" * 85)
print(
    f"{'Variable':<20}"
    f"{'Model 1':>15}"
    f"{'Model 2':>18}"
    f"{'LSTM':>18}"
)
print("-" * 85)

for variable in VARIABLES:

    print(
        f"{variable:<20}"
        f"{wasserstein_model1[variable]:>15.6f}"
        f"{wasserstein_model2[variable]:>18.6f}"
        f"{wasserstein_lstm[variable]:>18.6f}"
    )


# ============================================================
# PRINT STANDARDISED WASSERSTEIN
# ============================================================

print("\nStandardised Wasserstein Distance")
print("-" * 85)
print(
    f"{'Variable':<20}"
    f"{'Model 1':>15}"
    f"{'Model 2':>18}"
    f"{'LSTM':>18}"
)
print("-" * 85)

for variable in VARIABLES:

    print(
        f"{variable:<20}"
        f"{swasserstein_model1[variable]:>15.6f}"
        f"{swasserstein_model2[variable]:>18.6f}"
        f"{swasserstein_lstm[variable]:>18.6f}"
    )


# ============================================================
# PRINT KL
# ============================================================

print("\nGaussian KL Divergence (Supplementary)")
print("-" * 85)
print(
    f"{'Variable':<20}"
    f"{'Model 1':>15}"
    f"{'Model 2':>18}"
    f"{'LSTM':>18}"
)
print("-" * 85)

for variable in VARIABLES:

    print(
        f"{variable:<20}"
        f"{kl_model1[variable]:>15.6f}"
        f"{kl_model2[variable]:>18.6f}"
        f"{kl_lstm[variable]:>18.6f}"
    )


# ============================================================
# OVERALL STANDARDISED METRICS
# ============================================================

overall_smae_model1 = np.mean(
    list(smae_model1.values())
)

overall_smae_model2 = np.mean(
    list(smae_model2.values())
)

overall_smae_lstm = np.mean(
    list(smae_lstm.values())
)

overall_srmse_model1 = np.mean(
    list(srmse_model1.values())
)

overall_srmse_model2 = np.mean(
    list(srmse_model2.values())
)

overall_srmse_lstm = np.mean(
    list(srmse_lstm.values())
)

overall_swasserstein_model1 = np.mean(
    list(swasserstein_model1.values())
)

overall_swasserstein_model2 = np.mean(
    list(swasserstein_model2.values())
)

overall_swasserstein_lstm = np.mean(
    list(swasserstein_lstm.values())
)


# ============================================================
# PRINT OVERALL
# ============================================================

print("\nOverall Standardised Metrics")
print("-" * 60)

print(
    f"Model 1 Standardised MAE: "
    f"{overall_smae_model1:.6f}"
)

print(
    f"Model 2 Standardised MAE: "
    f"{overall_smae_model2:.6f}"
)

print(
    f"LSTM Standardised MAE: "
    f"{overall_smae_lstm:.6f}"
)

print()

print(
    f"Model 1 Standardised RMSE: "
    f"{overall_srmse_model1:.6f}"
)

print(
    f"Model 2 Standardised RMSE: "
    f"{overall_srmse_model2:.6f}"
)

print(
    f"LSTM Standardised RMSE: "
    f"{overall_srmse_lstm:.6f}"
)

print()

print(
    f"Model 1 Standardised Wasserstein: "
    f"{overall_swasserstein_model1:.6f}"
)

print(
    f"Model 2 Standardised Wasserstein: "
    f"{overall_swasserstein_model2:.6f}"
)

print(
    f"LSTM Standardised Wasserstein: "
    f"{overall_swasserstein_lstm:.6f}"
)