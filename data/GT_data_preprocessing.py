import pandas as pd

def load_data():
    return pd.read_pickle("data/Simulation/ground_truth_data.pkl")


def split_data(data):

    n_samples = len(data)

    train_end = int(n_samples * 0.70)
    validation_end = int(n_samples * 0.85)

    train_data = data.iloc[:train_end].copy()
    validation_data = data.iloc[train_end:validation_end].copy()
    test_data = data.iloc[validation_end:].copy()

    return train_data, validation_data, test_data


def save_data(train_data, validation_data, test_data):

    train_data.to_pickle("data/Simulation/GT_train.pkl")
    validation_data.to_pickle("data/Simulation/GT_validation.pkl")
    test_data.to_pickle("data/Simulation/GT_test.pkl")


if __name__ == "__main__":

    data = load_data()

    train_data, validation_data, test_data = split_data(data)

    save_data(
        train_data,
        validation_data,
        test_data
    )

    print("Total sequences:", len(data))
    print("Training sequences:", len(train_data))
    print("Validation sequences:", len(validation_data))
    print("Test sequences:", len(test_data))