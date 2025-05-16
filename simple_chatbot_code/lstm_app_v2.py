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

# إعداد الصفحة
st.set_page_config(page_title="Chatbot LSTM المطور", layout="wide", page_icon="🤖")

# تحميل نموذج اللغة الإنجليزية
@st.cache_resource
def load_spacy_model():
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        with st.spinner("جاري تحميل نموذج اللغة الإنجليزية (en_core_web_sm)... قد يستغرق هذا بعض الوقت."):
            spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    return nlp

nlp = load_spacy_model()

# المسارات
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
    pat_char = re.compile(r'[^A-Za-z]')
    pat_spaces = re.compile(r'\s+')

    msg = str(msg).lower()
    msg = msg.strip()
    msg = re.sub(pat_char,' ', msg)
    msg = re.sub(pat_spaces,' ', msg)

    tokens = nlp(msg)
    lemma = [token.lemma_ for token in tokens if not token.is_stop and not token.is_punct]
    cleaned_msg = " ".join(lemma).strip()

    return cleaned_msg

def predict_intent(text, model, tokenizer, lbl_encoder, max_len):
    cleaned_text = clean_pattern(text)
    
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, padding="post", maxlen=max_len)
    prediction = model.predict(padded_sequence)
    predicted_label_index = np.argmax(prediction, axis=1)
    predicted_tag = lbl_encoder.inverse_transform(predicted_label_index)[0]

    return predicted_tag

# تحسين تنسيق الرسائل
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stChatMessage[data-testid="chatAvatarIcon-user"] + div div {
        background-color: #dbeeff;
        border-left: 5px solid #3399ff;
    }
    .stChatMessage[data-testid="chatAvatarIcon-assistant"] + div div {
        background-color: #f2f2f2;
        border-right: 5px solid #999999;
    }
</style>
""", unsafe_allow_html=True)

# العنوان
st.title("🤖 Chatbot LSTM, DEPI CLS")
st.caption("مرحباً بك! أنا هنا لمساعدتك. كيف يمكنني خدمتك اليوم؟")

# الشريط الجانبي
with st.sidebar:
    st.header("عن الشات بوت")
    st.markdown("💡 هذا الشات بوت يستخدم نموذج LSTM للإجابة على استفساراتك.")
    st.markdown("🛠️ تم تطويره بواسطة فريق DEPI Team:")
    st.markdown("- Abdallah Samir\n- Youssef Samy\n- Shaaban Mosaad\n- Nada Amr\n- Mostafa Ahmed Elesely\n- Mohammed Ahmed Badrawy")

    if st.button("🧹 مسح سجل المحادثة"):
        st.session_state.messages = []

# تهيئة المحادثة
if "messages" not in st.session_state:
    st.session_state.messages = []

# عرض الرسائل السابقة
for message in st.session_state.messages:
    avatar_icon = "🧑‍💻" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar_icon):
        st.markdown(message["content"])

# مدخل الرسائل
if prompt := st.chat_input("💬 اكتب رسالتك هنا..."):
    # عرض رسالة المستخدم
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # الحصول على الرد من النموذج
    predicted_tag = predict_intent(prompt, model, tokenizer, lbl_encoder, max_len)

    response = "آسف، لم أفهم ذلك تمامًا. هل يمكنك إعادة صياغة سؤالك؟"
    if predicted_tag and predicted_tag in tags_answers:
        response = random.choice(tags_answers[predicted_tag])

    # عرض رد الشات بوت
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

