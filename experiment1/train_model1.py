import pandas as pd
import pickle 

from models.model1_DLGBN import model_1


def load_training_data():
    return pd.read_pickle("data/train.pkl")


def train_model():

    training_data = load_training_data()

    print("Training Model 1...")
    print("Training sequences:", len(training_data))

    model_1.fit(
        training_data,
        estimator="mle",
        std_estimator="unbiased"
    )
    with open("models/model1_trained.pkl", "wb") as f:
        pickle.dump(model_1, f)

    return model_1


if __name__ == "__main__":

    trained_model = train_model()

    print("\nModel 1 trained successfully.")

    for cpd in trained_model.get_cpds():
        print(cpd)