import streamlit as st
import json
import re
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import spacy
import random
import time

# Page config
st.set_page_config(page_title="Chatbot LSTM المطور", layout="wide", page_icon="🤖")

# Load English tokenizer, tagger, parser, NER and word vectors
@st.cache_resource
def load_spacy_model():
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        with st.spinner("جاري تحميل نموذج اللغة الإنجليزية..."):
            spacy.cli.download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
    return nlp

nlp = load_spacy_model()

# Paths
MODEL_PATH = "simple_chatbot_code/simple_chatbot_train_model.h5"
TOKENIZER_PATH = "simple_chatbot_code/tokenizer.pickle"
LABEL_ENCODER_PATH = "simple_chatbot_code/label_encoder.pickle"
MAX_LEN_PATH = "simple_chatbot_code/MAX_LEN.pickle"
TAGS_ANSWERS_PATH = "simple_chatbot_code/tags_answers.pickle"

@st.cache_resource
def load_all_resources():
    model = load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, "rb") as handle:
        tokenizer = pickle.load(handle)

    with open(LABEL_ENCODER_PATH, "rb") as handle:
        lbl_encoder = pickle.load(handle)

    with open(MAX_LEN_PATH, "rb") as handle:
        max_len = pickle.load(handle)

    with open(TAGS_ANSWERS_PATH, "rb") as handle:
        tags_answers = pickle.load(handle)

    return model, tokenizer, lbl_encoder, max_len, tags_answers

model, tokenizer, lbl_encoder, max_len, tags_answers = load_all_resources()

def clean_pattern(msg):
    msg = str(msg).lower().strip()
    msg = re.sub(r'[^A-Za-z]', ' ', msg)
    msg = re.sub(r'\s+', ' ', msg)
    tokens = nlp(msg)
    lemma = [token.lemma_ for token in tokens if not token.is_stop and not token.is_punct]
    return " ".join(lemma).strip()

def predict_intent(text, model, tokenizer, lbl_encoder, max_len):
    cleaned_text = clean_pattern(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, padding="post", maxlen=max_len)
    prediction = model.predict(padded_sequence, verbose=0)
    predicted_label_index = np.argmax(prediction, axis=1)
    predicted_tag = lbl_encoder.inverse_transform(predicted_label_index)[0]
    return predicted_tag

# --- UI Enhancements ---
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stChatMessage[data-testid="chatAvatarIcon-user"] + div div {
        background-color: #e6f3ff;
    }
    .stChatMessage[data-testid="chatAvatarIcon-assistant"] + div div {
        background-color: #f0f0f0;
    }
    .typing {
        color: #888;
        font-style: italic;
        animation: blink 1s step-start infinite;
    }
    @keyframes blink {
        50% { opacity: 0; }
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Chatbot LSTM, DEPI CLS")
st.caption("مرحباً بك! أنا هنا لمساعدتك. كيف يمكنني خدمتك اليوم؟")

# Sidebar
with st.sidebar:
    st.header("عن الشات بوت")
    st.markdown("هذا الشات بوت يستخدم نموذج LSTM للإجابة على استفساراتك.")
    st.markdown("تم تطويره بواسطة فريق DEPI Team.")
    st.markdown("* Abdallah Samir\n* Youssef Samy\n* Shaaban Mosaad\n* Nada Amr\n* Mohammed Ahmed Badrawy")
    
    st.markdown("---")
    tone = st.radio("اختر نغمة الردود:", ["رسمية", "ودية"], index=0)
    
    if st.button("مسح سجل المحادثة"):
        st.session_state.messages = []

    if st.session_state.get("messages"):
        st.download_button(
            label="📥 تحميل سجل المحادثة",
            data=json.dumps(st.session_state.messages, ensure_ascii=False, indent=2),
            file_name="chat_history.json",
            mime="application/json"
        )

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display message count
st.info(f"📨 عدد الرسائل في المحادثة: {len(st.session_state.messages)}")

# Display chat messages
for message in st.session_state.messages:
    avatar_icon = "🧑‍💻" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar_icon):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("اكتب رسالتك هنا..."):
    # User message
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Typing animation
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("يكتب..."):
            time.sleep(1.2)  # simulate delay

            predicted_tag = predict_intent(prompt, model, tokenizer, lbl_encoder, max_len)

            default_response = "آسف، لم أفهم ذلك تمامًا. حاول تشرح بطريقة مختلفة."
            response = default_response

            if predicted_tag in tags_answers:
                response = random.choice(tags_answers[predicted_tag])
                if tone == "ودية":
                    response = f"😊 {response}"
                else:
                    response = f"{response}."

            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

