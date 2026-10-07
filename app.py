import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import os

from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Emotion Detection",
    page_icon="😊",
    layout="wide"
)


# ============================================================
# EMOTION LABELS
# ============================================================

LABEL_NAMES = [
    "Sadness",
    "Joy",
    "Love",
    "Anger",
    "Fear",
    "Surprise"
]


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "fine_tuned_bilstm_emotion.keras"
)

TOKENIZER_PATH = os.path.join(
    BASE_DIR,
    "tokenizer",
    "tokenizer.pkl"
)


# ============================================================
# MAXIMUM SEQUENCE LENGTH
# ============================================================

# IMPORTANT:
# Use the SAME MAX_LEN that was used during model training.

MAX_LEN = 100


# ============================================================
# CHECK MODEL FILE
# ============================================================

if not os.path.exists(MODEL_PATH):

    st.error(
        "BiLSTM model file not found."
    )

    st.write(
        "Expected model path:"
    )

    st.code(
        MODEL_PATH
    )

    st.write(
        "Make sure this file exists:"
    )

    st.code(
        "models/fine_tuned_bilstm_emotion.keras"
    )

    st.stop()


# ============================================================
# CHECK TOKENIZER FILE
# ============================================================

if not os.path.exists(TOKENIZER_PATH):

    st.error(
        "Tokenizer file not found."
    )

    st.write(
        "Expected tokenizer path:"
    )

    st.code(
        TOKENIZER_PATH
    )

    st.write(
        "Make sure this file exists:"
    )

    st.code(
        "tokenizer/tokenizer.pkl"
    )

    st.stop()


# ============================================================
# LOAD TOKENIZER
# ============================================================

@st.cache_resource
def load_tokenizer():

    with open(
        TOKENIZER_PATH,
        "rb"
    ) as file:

        tokenizer = pickle.load(file)

    return tokenizer


# ============================================================
# LOAD BiLSTM MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model


# ============================================================
# LOAD MODEL AND TOKENIZER
# ============================================================

try:

    tokenizer = load_tokenizer()

    bilstm_model = load_model()

except Exception as e:

    st.error(
        "Error loading model or tokenizer."
    )

    st.exception(e)

    st.stop()


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_emotion(text):

    # Convert text into sequence
    sequences = tokenizer.texts_to_sequences(
        [text]
    )

    # Pad sequence
    padded_sequences = pad_sequences(
        sequences,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    # Model prediction
    prediction = bilstm_model.predict(
        padded_sequences,
        verbose=0
    )

    # Find predicted class
    predicted_class = np.argmax(
        prediction[0]
    )

    # Get confidence
    confidence = float(
        prediction[0][predicted_class]
    )

    # Get emotion name
    predicted_emotion = LABEL_NAMES[
        predicted_class
    ]

    return (
        predicted_emotion,
        confidence,
        prediction[0]
    )


# ============================================================
# TITLE
# ============================================================

st.title(
    "😊 Emotion Detection using Fine-Tuned BiLSTM"
)

st.write(
    """
This application uses a fine-tuned Bidirectional LSTM
(BiLSTM) model to classify emotions from text.
"""
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "Model Information"
)

st.sidebar.write(
    "Model: Fine-Tuned BiLSTM"
)

st.sidebar.write(
    "Framework: TensorFlow / Keras"
)

st.sidebar.write(
    "Task: Emotion Classification"
)

st.sidebar.write(
    "Number of Classes: 6"
)

st.sidebar.write(
    "Maximum Sequence Length:"
)

st.sidebar.write(
    MAX_LEN
)


# ============================================================
# TEXT INPUT
# ============================================================

st.subheader(
    "Enter Your Text"
)

text = st.text_area(
    "Type a sentence or paragraph:",
    placeholder=(
        "Example: I am very happy today "
        "because everything went well."
    ),
    height=180
)


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "🔍 Predict Emotion",
    use_container_width=True
):

    if text.strip() == "":

        st.warning(
            "Please enter some text."
        )

    else:

        emotion, confidence, probabilities = (
            predict_emotion(text)
        )

        # ----------------------------------------------------
        # PREDICTION RESULT
        # ----------------------------------------------------

        st.subheader(
            "Prediction Result"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.success(
                f"Predicted Emotion: {emotion}"
            )

        with col2:

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

        # ----------------------------------------------------
        # PROBABILITY DISTRIBUTION
        # ----------------------------------------------------

        st.subheader(
            "Emotion Probability Distribution"
        )

        probability_data = {
            LABEL_NAMES[i]: float(
                probabilities[i]
            )
            for i in range(
                len(LABEL_NAMES)
            )
        }

        st.bar_chart(
            probability_data
        )

        # ----------------------------------------------------
        # DETAILED PROBABILITIES
        # ----------------------------------------------------

        st.subheader(
            "Detailed Probabilities"
        )

        for i, emotion_name in enumerate(
            LABEL_NAMES
        ):

            probability = float(
                probabilities[i]
            )

            st.write(
                f"{emotion_name}: "
                f"{probability * 100:.2f}%"
            )

            st.progress(
                probability
            )


# ============================================================
# SUPPORTED EMOTIONS
# ============================================================

st.divider()

st.subheader(
    "Supported Emotions"
)

col1, col2, col3 = st.columns(3)

with col1:

    st.info(
        "😢 Sadness"
    )

    st.success(
        "😊 Joy"
    )


with col2:

    st.info(
        "❤️ Love"
    )

    st.warning(
        "😠 Anger"
    )


with col3:

    st.warning(
        "😨 Fear"
    )

    st.success(
        "😲 Surprise"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "NLP Emotion Detection | "
    "Fine-Tuned BiLSTM | "
    "TensorFlow + Streamlit"
)
