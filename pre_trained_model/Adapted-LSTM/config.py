class Config:
    # data
    window_size = 100
    stride = 10
    # SKAB CSV uses ';' separator and has:
    # datetime + 8 sensor features + anomaly + changepoint
    label_col = "anomaly"
    time_col = "datetime"
    feature_cols = [
        "Accelerometer1RMS",
        "Accelerometer2RMS",
        "Current",
        "Pressure",
        "Temperature",
        "Thermocouple",
        "Voltage",
        "Volume Flow RateRMS",
    ]

    # model
    input_dim = len(feature_cols)
    hidden_dim = 64
    latent_dim = 32

    # training
    batch_size = 64
    lr = 1e-3
    epochs = 20
    # evaluation / thresholding
    threshold_quantile = 0.95  # use normal-only val scores
    early_stop_patience = 5

    # SVDD
    margin = 2.0
    lambda_anomaly = 1.0

    # contrastive
    lambda_contrast = 0.5
    temperature = 0.2