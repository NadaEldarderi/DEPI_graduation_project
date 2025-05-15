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

# Page config
st.set_page_config(page_title="Chatbot LSTM المطور", layout="wide", page_icon="🤖")

# Load English tokenizer, tagger, parser, NER and word vectors
@st.cache_resource
def load_spacy_model():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        with st.spinner("جاري تحميل نموذج اللغة الإنجليزية (en_core_web_sm)... قد يستغرق هذا بعض الوقت."):
            spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    return nlp

nlp = load_spacy_model()

# Load pre-trained model and other necessary files
MODEL_PATH = "/home/ubuntu/lstm_project/simple_chatbot_code/simple_chatbot_train_model.h5"
TOKENIZER_PATH = "/home/ubuntu/lstm_project/simple_chatbot_code/tokenizer.pickle"
LABEL_ENCODER_PATH = "/home/ubuntu/lstm_project/simple_chatbot_code/label_encoder.pickle"
MAX_LEN_PATH = "/home/ubuntu/lstm_project/simple_chatbot_code/MAX_LEN.pickle"
TAGS_ANSWERS_PATH = "/home/ubuntu/lstm_project/simple_chatbot_code/tags_answers.pickle"

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
    pat_char = re.compile(r"[^A-Za-z0-9\sآء-ي]") # Allow spaces, numbers, and Arabic characters
    pat_spaces = re.compile(r"\s+")
    msg = str(msg).lower()
    msg = msg.strip()
    msg = re.sub(pat_char, "", msg)
    msg = re.sub(pat_spaces, " ", msg)
    # For English text, lemmatize. For Arabic, basic cleaning is done.
    # A more sophisticated Arabic NLP library would be needed for proper Arabic lemmatization.
    if any(char.isalpha() and ord(char) < 128 for char in msg): # Basic check for English text
        tokens = nlp(msg)
        lemma = [token.lemma_ for token in tokens if not token.is_stop and not token.is_punct]
        cleaned_msg = " ".join(lemma).strip()
    else: # Assume Arabic or mixed, apply basic cleaning
        cleaned_msg = msg
    return cleaned_msg

def predict_intent(text, model, tokenizer, lbl_encoder, max_len):
    cleaned_text = clean_pattern(text)
    if not cleaned_text:
        return None
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, truncating="post", maxlen=max_len)
    prediction = model.predict(padded_sequence)
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
        background-color: #e6f3ff; /* Light blue for user messages */
    }
    .stChatMessage[data-testid="chatAvatarIcon-assistant"] + div div {
        background-color: #f0f0f0; /* Light grey for assistant messages */
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Chatbot LSTM المطور")
st.caption("مرحباً بك! أنا هنا لمساعدتك. كيف يمكنني خدمتك اليوم؟")

# Sidebar
with st.sidebar:
    st.header("عن الشات بوت")
    st.markdown("هذا الشات بوت يستخدم نموذج LSTM للإجابة على استفساراتك.")
    st.markdown("تم تطويره بواسطة فريق Manus.")
    if st.button("مسح سجل المحادثة"):
        st.session_state.messages = []

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    avatar_icon = "🧑‍💻" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar_icon):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("اكتب رسالتك هنا..."):
    # Display user message in chat message container
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    predicted_tag = predict_intent(prompt, model, tokenizer, lbl_encoder, max_len)
    
    response = "آسف، لم أفهم ذلك تمامًا. هل يمكنك محاولة إعادة صياغة سؤالك أو طرح سؤال آخر؟"
    if predicted_tag and predicted_tag in tags_answers:
        possible_responses = tags_answers[predicted_tag]
        response = random.choice(possible_responses)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

