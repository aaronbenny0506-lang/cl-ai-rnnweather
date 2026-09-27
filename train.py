"""
Task 10: Weather Prediction using RNN
--------------------------------------
Trains an LSTM-based RNN on the London Weather Data (london_weather.csv) to
predict next-day mean, max and min temperature and precipitation from the
past WINDOW days of weather.

Usage:
    python train.py --data data/london_weather.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

FEATURES = [
    "cloud_cover", "sunshine", "global_radiation", "max_temp", "mean_temp",
    "min_temp", "precipitation", "pressure", "snow_depth",
]
TARGETS = ["mean_temp", "max_temp", "min_temp", "precipitation"]
WINDOW = 14          # days of history used to predict the next day
TEST_FRACTION = 0.15
SEED = 42


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.sort_values("date").reset_index(drop=True)
    # linear interpolation handles the small gaps present in the raw ECA data
    df[FEATURES] = df[FEATURES].interpolate(method="linear", limit_direction="both")
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    return df


def make_sequences(arr: np.ndarray, window: int, target_cols: list):
    X, y = [], []
    for i in range(len(arr) - window):
        X.append(arr[i:i + window])
        y.append(arr[i + window, target_cols])
    return np.array(X), np.array(y)


def build_model(window: int, n_features: int, n_targets: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(window, n_features)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dense(32, activation="relu"),
        layers.Dense(n_targets),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss="mse", metrics=["mae"])
    return model


def main(args):
    np.random.seed(SEED)
    df = load_and_clean(args.data)

    n = len(df)
    test_start = int(n * (1 - TEST_FRACTION))
    train_df, test_df = df.iloc[:test_start], df.iloc[test_start:]

    # fit the scaler on the training split only, to avoid leaking test
    # statistics into training (important for time series data)
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df[FEATURES])
    test_scaled = scaler.transform(test_df[FEATURES])

    feat_idx = {f: i for i, f in enumerate(FEATURES)}
    target_cols = [feat_idx[t] for t in TARGETS]

    X_train, y_train = make_sequences(train_scaled, WINDOW, target_cols)
    X_test, y_test = make_sequences(test_scaled, WINDOW, target_cols)
    print(f"Train sequences: {X_train.shape}, Test sequences: {X_test.shape}")

    model = build_model(WINDOW, len(FEATURES), len(TARGETS))
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                       restore_best_weights=True),
    ]
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=2,
    )

    y_pred_scaled = model.predict(X_test)

    # inverse-transform predictions and targets back to real units
    def inverse(y_scaled):
        dummy = np.zeros((len(y_scaled), len(FEATURES)))
        dummy[:, target_cols] = y_scaled
        return scaler.inverse_transform(dummy)[:, target_cols]

    y_test_real = inverse(y_test)
    y_pred_real = inverse(y_pred_scaled)

    os.makedirs("outputs", exist_ok=True)

    print("\nPer-target test metrics:")
    metrics = {}
    for i, t in enumerate(TARGETS):
        mse = np.mean((y_test_real[:, i] - y_pred_real[:, i]) ** 2)
        mae = np.mean(np.abs(y_test_real[:, i] - y_pred_real[:, i]))
        metrics[t] = {"mse": mse, "mae": mae}
        print(f"  {t:15s} MSE={mse:.3f}  MAE={mae:.3f}")

    pd.DataFrame(metrics).T.to_csv("outputs/metrics.csv")

    # plot actual vs predicted for mean_temp (the primary target)
    plt.figure(figsize=(10, 4))
    plt.plot(y_test_real[:, 0], label="Actual mean_temp")
    plt.plot(y_pred_real[:, 0], label="Predicted mean_temp")
    plt.xlabel("Test day index")
    plt.ylabel("Mean temperature (°C)")
    plt.title("Actual vs Predicted Next-Day Mean Temperature")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/actual_vs_predicted.png", dpi=150)
    print("\nSaved outputs/metrics.csv and outputs/actual_vs_predicted.png")

    model.save("outputs/weather_rnn.keras")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/london_weather.csv")
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()
    main(args)
