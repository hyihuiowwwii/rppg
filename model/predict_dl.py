import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("bpm_model.h5")

def predict_bpm_from_signal(signal_window):
    signal_window = np.array(signal_window, dtype=np.float32)

    if len(signal_window) < 150:
        return 0

    # Take last 150 values
    signal_window = signal_window[-150:]

    # Reshape for model
    signal_window = signal_window.reshape(1, 150, 1)

    # Predict BPM
    prediction = model.predict(signal_window, verbose=0)

    return float(prediction[0][0])
