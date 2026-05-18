# ============================================================
# SKAB LSTM Autoencoder - Data Cleaning & Preparation
# Train  : anomaly-free/anomaly-free.csv
# Test   : valve1/*.csv, valve2/*.csv, other/*.csv
# Author : Ready for Google Colab
# ============================================================

import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1. Paths
# -----------------------------
BASE_DIR = "/Users/thantsinoo/Python/SKAB"
TRAIN_PATH = os.path.join(BASE_DIR, "anomaly-free", "anomaly-free.csv")

TEST_DIRS = {
    "valve1": os.path.join(BASE_DIR, "valve1"),
    "valve2": os.path.join(BASE_DIR, "valve2"),
    "other": os.path.join(BASE_DIR, "other")
}

# -----------------------------
# 2. Feature columns
# -----------------------------
BASE_FEATURES = [
    "Accelerometer1RMS",
    "Accelerometer2RMS",
    "Current",
    "Pressure",
    "Temperature",
    "Thermocouple",
    "Voltage",
    "Volume Flow RateRMS"
]

LABEL_COL = "anomaly"
TIME_COL = "datetime"

ROLLING_WINDOW = 5
SEQUENCE_LENGTH = 30

# -----------------------------
# 3. Helper functions
# -----------------------------
def clean_column_names(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df


def parse_datetime_column(df, time_col=TIME_COL):
    df = df.copy()
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col])
        df = df.sort_values(time_col).reset_index(drop=True)
    return df


def convert_feature_columns_to_numeric(df, feature_cols):
    df = df.copy()
    for col in feature_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_rolling_features(df, feature_cols, window=5):
    df = df.copy()

    for col in feature_cols:
        df[f"{col}_roll_mean"] = df[col].rolling(window=window, min_periods=window).mean()
        df[f"{col}_roll_std"] = df[col].rolling(window=window, min_periods=window).std()

    df = df.dropna().reset_index(drop=True)
    return df


def basic_clean(df, feature_cols, label_col=None, time_col=TIME_COL, rolling_window=5):
    """
    Clean a SKAB dataframe:
    - strip column names
    - parse datetime
    - keep only needed columns
    - convert features to numeric
    - handle label if present
    - drop missing rows
    - add rolling features
    """
    df = clean_column_names(df)
    df = parse_datetime_column(df, time_col=time_col)

    required_cols = []
    if time_col in df.columns:
        required_cols.append(time_col)

    required_cols += [col for col in feature_cols if col in df.columns]

    if label_col is not None and label_col in df.columns:
        required_cols.append(label_col)

    df = df[required_cols].copy()

    # convert base features to numeric
    df = convert_feature_columns_to_numeric(df, feature_cols)

    # label handling for test files
    if label_col is not None and label_col in df.columns:
        df[label_col] = pd.to_numeric(df[label_col], errors="coerce")

    # drop rows with missing values before rolling
    cols_to_check = [col for col in feature_cols if col in df.columns]
    if label_col is not None and label_col in df.columns:
        cols_to_check.append(label_col)

    df = df.dropna(subset=cols_to_check).reset_index(drop=True)

    # add rolling features
    df = add_rolling_features(df, feature_cols, window=rolling_window)

    return df


def get_all_feature_columns(df, base_features):
    """
    Return base features + generated rolling features
    """
    return [col for col in df.columns if col in base_features or "_roll_" in col]


def create_sequences(data, seq_len):
    """
    Convert 2D array -> 3D array for LSTM
    shape: (num_sequences, seq_len, num_features)
    """
    sequences = []
    for i in range(len(data) - seq_len + 1):
        sequences.append(data[i:i + seq_len])
    return np.array(sequences)


def create_sequence_labels(labels, seq_len):
    """
    For each sequence, use the last timestep label
    """
    seq_labels = []
    for i in range(len(labels) - seq_len + 1):
        seq_labels.append(labels[i + seq_len - 1])
    return np.array(seq_labels)


def load_and_prepare_train(train_path, base_features, rolling_window=5, seq_len=30):
    print(f"Loading training file: {train_path}")
    train_df = pd.read_csv(train_path, sep=";")

    train_df = basic_clean(
        train_df,
        feature_cols=base_features,
        label_col=None,
        time_col=TIME_COL,
        rolling_window=rolling_window
    )

    all_features = get_all_feature_columns(train_df, base_features)

    # fit scaler ONLY on train
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df[all_features])

    # create LSTM sequences
    X_train = create_sequences(train_scaled, seq_len)

    print("\n===== TRAIN DATA =====")
    print("Train dataframe shape:", train_df.shape)
    print("Number of training features:", len(all_features))
    print("X_train shape:", X_train.shape)

    return train_df, all_features, scaler, X_train


def load_and_prepare_test_folder(folder_path, group_name, base_features, all_features, scaler,
                                 rolling_window=5, seq_len=30):
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))

    group_results = []

    print(f"\nLoading test folder: {group_name}")
    print(f"Number of files: {len(csv_files)}")

    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path, sep=";")
        except Exception:
            # fallback in case some file uses comma
            df = pd.read_csv(file_path)

        df = basic_clean(
            df,
            feature_cols=base_features,
            label_col=LABEL_COL,
            time_col=TIME_COL,
            rolling_window=rolling_window
        )

        # make sure all expected columns exist
        missing_features = [col for col in all_features if col not in df.columns]
        if missing_features:
            print(f"Skipping {file_path} بسبب missing features: {missing_features}")
            continue

        X_scaled = scaler.transform(df[all_features])
        X_seq = create_sequences(X_scaled, seq_len)

        y_seq = None
        if LABEL_COL in df.columns:
            y_seq = create_sequence_labels(df[LABEL_COL].values, seq_len)

        result = {
            "group": group_name,
            "file_name": os.path.basename(file_path),
            "clean_df": df,
            "X_scaled": X_scaled,
            "X_seq": X_seq,
            "y_seq": y_seq
        }

        group_results.append(result)

        print(f"Processed: {os.path.basename(file_path)}")
        print(f"  Clean df shape : {df.shape}")
        print(f"  X_seq shape    : {X_seq.shape}")
        if y_seq is not None:
            print(f"  y_seq shape    : {y_seq.shape}")
            print(f"  anomaly count  : {int(np.sum(y_seq))}")

    return group_results


# -----------------------------
# 4. Prepare training data
# -----------------------------
train_df, all_features, scaler, X_train = load_and_prepare_train(
    train_path=TRAIN_PATH,
    base_features=BASE_FEATURES,
    rolling_window=ROLLING_WINDOW,
    seq_len=SEQUENCE_LENGTH
)

# -----------------------------
# 5. Prepare test data
# -----------------------------
test_results = {}

for group_name, folder_path in TEST_DIRS.items():
    test_results[group_name] = load_and_prepare_test_folder(
        folder_path=folder_path,
        group_name=group_name,
        base_features=BASE_FEATURES,
        all_features=all_features,
        scaler=scaler,
        rolling_window=ROLLING_WINDOW,
        seq_len=SEQUENCE_LENGTH
    )

# Step for model  training LSTM.
# loading data
# cleaning data
# rolling features
# scaling
# sequence creation

# Now the remaining steps are:
# LSTM Autoencoder model
# training
# reconstruction
# thresholding
# evaluation

# ============================================================
# 6. Imports for LSTM Autoencoder
# ============================================================
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt


# ============================================================
# 7. Build LSTM Autoencoder
# ============================================================
def build_lstm_autoencoder(seq_len, num_features, latent_dim=16):
    inputs = Input(shape=(seq_len, num_features))

    # Encoder
    x = LSTM(64, activation="tanh", return_sequences=True)(inputs)
    x = LSTM(latent_dim, activation="tanh", return_sequences=False)(x)

    # Bottleneck repeated across timesteps
    x = RepeatVector(seq_len)(x)

    # Decoder
    x = LSTM(latent_dim, activation="tanh", return_sequences=True)(x)
    x = LSTM(64, activation="tanh", return_sequences=True)(x)

    outputs = TimeDistributed(Dense(num_features))(x)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


SEQ_LEN = X_train.shape[1]
NUM_FEATURES = X_train.shape[2]

model = build_lstm_autoencoder(
    seq_len=SEQ_LEN,
    num_features=NUM_FEATURES,
    latent_dim=16
)

print(model.summary())


# ============================================================
# 8. Train / Validation split from X_train
# ============================================================
split_idx = int(len(X_train) * 0.8)

X_train_fit = X_train[:split_idx]
X_val_fit = X_train[split_idx:]

print("X_train_fit shape:", X_train_fit.shape)
print("X_val_fit shape:", X_val_fit.shape)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train_fit,
    X_train_fit,
    validation_data=(X_val_fit, X_val_fit),
    epochs=30,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)


# ============================================================
# 9. Plot training history
# ============================================================
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("LSTM Autoencoder Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# 10. Reconstruction error function
# ============================================================
def reconstruction_errors(model, X):
    X_pred = model.predict(X, verbose=0)
    errors = np.mean(np.square(X - X_pred), axis=(1, 2))
    return errors, X_pred


# ============================================================
# 11. Choose threshold from training reconstruction error
# ============================================================
train_errors, X_train_pred = reconstruction_errors(model, X_train)

print("Train reconstruction error stats:")
print("Min   :", np.min(train_errors))
print("Mean  :", np.mean(train_errors))
print("Median:", np.median(train_errors))
print("Max   :", np.max(train_errors))

# Start with 95th percentile
threshold = np.percentile(train_errors, 95)

print("\nChosen threshold (95th percentile of train error):", threshold)

plt.figure(figsize=(8, 5))
plt.hist(train_errors, bins=50)
plt.axvline(threshold, linestyle="--", label=f"Threshold = {threshold:.6f}")
plt.title("Training Reconstruction Error Distribution")
plt.xlabel("Reconstruction Error")
plt.ylabel("Count")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# 12. Evaluate each test file
# ============================================================
all_y_true = []
all_y_pred = []

file_level_results = []

for group_name, group_items in test_results.items():
    print("\n" + "=" * 60)
    print(f"GROUP: {group_name}")
    print("=" * 60)

    for item in group_items:
        file_name = item["file_name"]
        X_test_seq = item["X_seq"]
        y_true = item["y_seq"]

        if X_test_seq is None or len(X_test_seq) == 0:
            print(f"Skipping {file_name} because X_seq is empty.")
            continue

        if y_true is None or len(y_true) == 0:
            print(f"Skipping {file_name} because y_seq is empty.")
            continue

        test_errors, X_test_pred = reconstruction_errors(model, X_test_seq)
        y_pred = (test_errors > threshold).astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        print(f"\nFile: {file_name}")
        print("  X_test_seq shape:", X_test_seq.shape)
        print("  y_true shape    :", y_true.shape)
        print("  Pred anomalies  :", int(np.sum(y_pred)))
        print("  True anomalies  :", int(np.sum(y_true)))
        print(f"  Precision       : {precision:.4f}")
        print(f"  Recall          : {recall:.4f}")
        print(f"  F1-score        : {f1:.4f}")
        print("  Confusion Matrix:")
        print(cm)

        file_level_results.append({
            "group": group_name,
            "file_name": file_name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predicted_anomalies": int(np.sum(y_pred)),
            "true_anomalies": int(np.sum(y_true))
        })

        all_y_true.append(y_true)
        all_y_pred.append(y_pred)


# ============================================================
# 13. Overall evaluation
# ============================================================
if len(all_y_true) > 0 and len(all_y_pred) > 0:
    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)

    overall_precision = precision_score(all_y_true, all_y_pred, zero_division=0)
    overall_recall = recall_score(all_y_true, all_y_pred, zero_division=0)
    overall_f1 = f1_score(all_y_true, all_y_pred, zero_division=0)
    overall_cm = confusion_matrix(all_y_true, all_y_pred)

    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    print(f"Overall Precision : {overall_precision:.4f}")
    print(f"Overall Recall    : {overall_recall:.4f}")
    print(f"Overall F1-score  : {overall_f1:.4f}")
    print("Overall Confusion Matrix:")
    print(overall_cm)

    print("\nClassification Report:")
    print(classification_report(all_y_true, all_y_pred, digits=4, zero_division=0))

else:
    print("No valid test predictions were collected.")


# ============================================================
# 14. Save file-level results to CSV
# ============================================================
results_df = pd.DataFrame(file_level_results)
results_df.to_csv("lstm_file_level_results.csv", index=False)

print("\nSaved file-level results to lstm_file_level_results.csv")
print(results_df.head())