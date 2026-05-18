"""Shared feature and model hyper-parameters for UI inference."""

FEATURE_COLS = [
    "Accelerometer1RMS", "Accelerometer2RMS", "Current",
    "Pressure", "Temperature", "Thermocouple",
    "Voltage", "Volume Flow RateRMS",
]

EXTENDED_COLS = (
    FEATURE_COLS
    + [f"{c}_roll_mean" for c in FEATURE_COLS]
    + [f"{c}_roll_std" for c in FEATURE_COLS]
)

LSTM_SEQ_LEN = 30
LSTM_LATENT = 16

TRANSFORMER_SEQ_LEN = 30
TRANSFORMER_D_MODEL = 32
TRANSFORMER_NHEAD = 2
TRANSFORMER_LAYERS = 1
TRANSFORMER_FFN_DIM = 64

ALSTM_WINDOW = 100
ALSTM_STRIDE = 10
ALSTM_HIDDEN = 64
ALSTM_LATENT = 32
