import os
import numpy as np
import librosa
import soundfile as sf
import streamlit as st
import tempfile
import sounddevice as sd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# STEP 1: Simulate tiny dataset using tone audio (for demo only)
def generate_tone(freq, duration=3, sr=16000):
    t = np.linspace(0, duration, int(sr*duration), False)
    tone = 0.5 * np.sin(2 * np.pi * freq * t)
    return tone

def create_dataset():
    X, y = [], []
    # Simulate 3 Indian accents using 440Hz tone (low pitch)
    for _ in range(3):
        tone = generate_tone(440)
        mfcc = extract_mfcc(tone)
        X.append(mfcc)
        y.append("indian")
    # Simulate 3 American accents using 880Hz tone (high pitch)
    for _ in range(3):
        tone = generate_tone(880)
        mfcc = extract_mfcc(tone)
        X.append(mfcc)
        y.append("american")
    return np.array(X), np.array(y)

# STEP 2: Extract MFCC
def extract_mfcc(audio, sr=16000):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc = np.pad(mfcc, ((0,0),(0, max(0, 130 - mfcc.shape[1]))), mode='constant')[:, :130]
    return mfcc

# STEP 3: Train model
def train_model():
    X, y = create_dataset()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    y_cat = to_categorical(y_enc)

    X = X.reshape(-1, 40, 130, 1)

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(40,130,1)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(le.classes_), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y_cat, epochs=10, batch_size=2, verbose=0)

    model.save("cnn_model.h5")
    np.save("label_encoder.npy", le.classes_)

    return model, le.classes_

# STEP 4: Streamlit UI
st.title("🎙️ Accent Classifier: India 🇮🇳 vs America 🇺🇸")
st.markdown("Click record, speak for 3 seconds, and we'll guess your accent!")

# Train once and load model
if not os.path.exists("cnn_model.h5"):
    st.warning("Training model... please wait")
    model, label_classes = train_model()
else:
    model = load_model("cnn_model.h5")
    label_classes = np.load("label_encoder.npy", allow_pickle=True)

# Record voice
def record_voice(duration=3, fs=16000):
    st.info("Recording for 3 seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    st.success("Recording complete!")
    return np.squeeze(audio)

# Inference
if st.button("🎙️ Record & Predict"):
    audio = record_voice()
    st.audio(audio, format='audio/wav', sample_rate=16000)

    mfcc = extract_mfcc(audio).reshape(1, 40, 130, 1)
    pred = model.predict(mfcc)
    label = label_classes[np.argmax(pred)]
    st.success(f"🌍 Predicted Accent: **{label.upper()}**")
    st.write("📊 Prediction Probabilities:", {label_classes[i]: round(float(p), 2) for i, p in enumerate(pred[0])})
