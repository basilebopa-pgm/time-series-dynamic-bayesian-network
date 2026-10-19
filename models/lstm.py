import tensorflow as tf


def create_lstm(input_shape, output_size):

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(output_size)
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model