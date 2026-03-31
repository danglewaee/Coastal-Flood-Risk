import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, GlobalAveragePooling1D, Input, LayerNormalization, MultiHeadAttention, Add
from tensorflow.keras.optimizers import Adam


def weighted_peak_mse(peak_threshold: float, alpha: float, temperature: float):
    """Higher loss weight for targets in extreme-water tail."""

    threshold = tf.constant(peak_threshold, dtype=tf.float32)
    alpha_c = tf.constant(alpha, dtype=tf.float32)
    temp_c = tf.constant(max(temperature, 1e-5), dtype=tf.float32)

    def loss(y_true, y_pred):
        sq_err = tf.square(y_true - y_pred)
        weights = 1.0 + alpha_c * tf.sigmoid((y_true - threshold) / temp_c)
        return tf.reduce_mean(weights * sq_err)

    return loss


def build_lstm_model(
    lookback: int,
    n_features: int = 1,
    hidden_units: int = 64,
    lstm_layers: int = 2,
    dropout: float = 0.15,
    learning_rate: float = 1e-3,
):
    model = Sequential(name="sea_level_lstm")
    for i in range(lstm_layers):
        return_sequences = i < lstm_layers - 1
        layer_kwargs = {
            "units": hidden_units,
            "return_sequences": return_sequences,
            "activation": "tanh",
        }
        if i == 0:
            layer_kwargs["input_shape"] = (lookback, n_features)
        model.add(LSTM(**layer_kwargs))
        if dropout > 0:
            model.add(Dropout(dropout))

    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    return model


def build_temporal_cnn_model(
    lookback: int,
    n_features: int = 1,
    hidden_units: int = 64,
    dropout: float = 0.15,
    learning_rate: float = 1e-3,
):
    inputs = Input(shape=(lookback, n_features), name="input_series")
    x = Conv1D(filters=hidden_units, kernel_size=3, padding="causal", activation="relu")(inputs)
    x = Conv1D(filters=hidden_units, kernel_size=5, padding="causal", activation="relu", dilation_rate=2)(x)
    x = Conv1D(filters=hidden_units // 2, kernel_size=3, padding="causal", activation="relu", dilation_rate=4)(x)
    if dropout > 0:
        x = Dropout(dropout)(x)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(1, name="sea_level_next")(x)
    model = Model(inputs=inputs, outputs=outputs, name="sea_level_temporal_cnn")
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    return model


def build_axial_lstm_model(
    lookback: int,
    n_features: int = 1,
    hidden_units: int = 64,
    heads: int = 4,
    dropout: float = 0.15,
    learning_rate: float = 1e-3,
):
    inputs = Input(shape=(lookback, n_features), name="input_series")
    x = LSTM(hidden_units, return_sequences=True, activation="tanh")(inputs)

    attn_in = LayerNormalization()(x)
    attn = MultiHeadAttention(num_heads=heads, key_dim=max(8, hidden_units // heads), dropout=dropout)(attn_in, attn_in)
    x = Add()([x, attn])
    x = LayerNormalization()(x)

    x = LSTM(hidden_units // 2, return_sequences=False, activation="tanh")(x)
    if dropout > 0:
        x = Dropout(dropout)(x)
    outputs = Dense(1, name="sea_level_next")(x)

    model = Model(inputs=inputs, outputs=outputs, name="sea_level_axial_lstm")
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    return model


def build_model(model_type: str, lookback: int, hidden_units: int, lstm_layers: int, dropout: float, learning_rate: float, n_features: int = 1):
    model_key = model_type.lower().strip()
    if model_key == "lstm":
        return build_lstm_model(lookback, n_features, hidden_units, lstm_layers, dropout, learning_rate)
    if model_key == "temporal_cnn":
        return build_temporal_cnn_model(lookback, n_features, hidden_units, dropout, learning_rate)
    if model_key == "axial_lstm":
        return build_axial_lstm_model(lookback, n_features, hidden_units, 4, dropout, learning_rate)
    raise ValueError(f"Unsupported model_type: {model_type}")
