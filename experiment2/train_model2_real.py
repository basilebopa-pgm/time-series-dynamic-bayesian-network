import pandas as pd
import pickle

from models.model2_seasonal import model_2


def load_training_data():
    return pd.read_pickle(
        "data/real_train.pkl"
    )


def train_model():

    training_data = load_training_data()

    print("Training Model 2...")
    print(
        "Training sequences:",
        len(training_data)
    )

    model_2.fit(
        training_data,
        estimator="mle",
        std_estimator="unbiased"
    )

    with open(
        "models/model2_real_trained.pkl",
        "wb"
    ) as f:

        pickle.dump(
            model_2,
            f
        )

    return model_2


if __name__ == "__main__":

    trained_model = train_model()

    print(
        "\nModel 2 trained successfully."
    )

    for cpd in trained_model.get_cpds():

        print(cpd)