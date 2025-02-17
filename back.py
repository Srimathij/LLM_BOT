import streamlit as st
import os
import uuid
import base64
import json
import speech_recognition as sr
from audio_recorder_streamlit import audio_recorder
from streamlit_float import *
from dotenv import load_dotenv
import cohere
# from groq import Groq
import librosa
import soundfile as sf
from langdetect import detect
import torch
from elevenlabs import ElevenLabs

# Load environment variables
load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")

# Initialize Cohere client
co = cohere.Client(api_key=cohere_api_key)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Initialize Eleven Labs client
eleven_client = ElevenLabs(
    api_key="sk_180310ac98502229e8f63b6353f57e07e82b4c73d4a567c2",
)

# Function to generate speech in Malay or English using Eleven Labs

def text_to_speech(input_text, language='en', voice_id='21m00Tcm4TlvDq8ikWAM'):
    if language == 'id':
        # Optional: Use a specific Malay-compatible voice if available
        voice_id = '21m00Tcm4TlvDq8ikWAM'  # Replace with specific voice if needed
    
    # Generate audio from Eleven Labs
    audio_generator = eleven_client.text_to_speech.convert(
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",
        text=input_text,
    )

    # Save the audio as a file
    audio_file_path = f"{uuid.uuid4()}.mp3"
    with open(audio_file_path, "wb") as file:
        for chunk in audio_generator:
            file.write(chunk)

    return audio_file_path

# Function to autoplay audio in Streamlit
def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    md = f"""
    <audio autoplay>
    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(md, unsafe_allow_html=True)


# Preprocess audio function
def preprocess_audio(audio_path):
    # Load the audio file using librosa
    audio, sr = librosa.load(audio_path, sr=16000)

    # Save preprocessed audio to a temporary file
    temp_audio_path = f"temp_{uuid.uuid4()}.wav"
    sf.write(temp_audio_path, audio, sr)  # Save using soundfile

    return temp_audio_path

@st.cache_data
def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Load rooms.json
json_file_path = "rooms.json"
rooms_data = load_json(json_file_path)

def search_rooms(location, budget_range):
    min_budget, max_budget = budget_range
    results = [room for room in rooms_data if room['Location'].lower() == location.lower() and min_budget <= room['Budget'] <= max_budget]
    return results

# Function to get a response from Cohere LLM
def get_answer(chat_history, user_query, user_location=None, user_budget=None):
    prompt_template = f"""
        ### Chat History:
        {chat_history}

        ### User Question:
        {user_query}

        ### Instructions for Assistant:
        You are Telekom Malaysia's official voice bot, tasked with providing accurate and helpful information about Telekom Malaysia's products, services, and general inquiries. Always answer questions professionally and clearly, maintaining a friendly tone. 
    """
    
    try:
        stream = co.chat_stream(
            model='command-r-08-2024',
            message=prompt_template,
            temperature=0.3,
            chat_history=[],
            prompt_truncation='AUTO'
        )

        response_text = ""
        for event in stream:
            if event.event_type == "text-generation":
                response_text += event.text
        
        return response_text.strip()
    
    except Exception as e:
        print("Error occurred:", e)
        return "An error occurred while processing your request."

# import speech_recognition as sr
from langdetect import detect

def speech_to_text(audio_path):
    recognizer = sr.Recognizer()
    
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)

    try:
        recognized_text = recognizer.recognize_google(audio_data)  # Using Google's STT
        detected_language = detect(recognized_text)

        if detected_language not in ['en', 'id']:  # Restrict to English & Malay
            print("Unsupported language detected.")
            return "", "unsupported"

        print(f"User said: {recognized_text}")
        return recognized_text, detected_language

    except Exception as e:
        print(f"Speech recognition failed: {e}")
        return "", "unknown"

float_init()

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello and welcome to Telekom Malaysia's AI! 🌟 Your personal assistant is here to make your day easier, smarter, and more connected. How can I help you today?"}
        ]
    if "user_query" not in st.session_state:
        st.session_state.user_query = ""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "user_location" not in st.session_state:
        st.session_state.user_location = None
    if "user_budget" not in st.session_state:
        st.session_state.user_budget = (0, float("inf"))

initialize_session_state()

# Center the header with refined styles using Streamlit's HTML rendering
st.markdown("""
    <style>
    .header {
        text-align: center;
        font-size: 2rem;  /* Medium size header */
        font-weight: normal;  /* No bold */
        margin-top: 15px;
        color: linear-gradient(135deg, #f3ec78, #af4261); /* Gradient for contrast */
    }
    .mic-container {
        position: fixed;
        bottom: 1rem;
        right: 1rem;
    }
    body {
        background-color: #000; /* Black background */
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='header'>VoiceBot.AI✨</div>", unsafe_allow_html=True)

# Create footer container for the microphone
footer_container = st.container()

# Initialize the user's query to None
user_query = None

# Handle audio input in the footer container
with footer_container:
    transcript = None

    audio_bytes = audio_recorder(
        text="",
        recording_color="#e8b62c",
        neutral_color="#6aa36f",
        icon_name="microphone-lines",
        icon_size="3x",
    )

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if audio_bytes:
    with st.spinner("Hang on! I’m finding the best options for you...⏳"):
        webm_file_path = "temp_audio.mp3"
        with open(webm_file_path, "wb") as f:
            f.write(audio_bytes)
        transcript, detected_language = speech_to_text(webm_file_path)
        
        if transcript:
            st.session_state.messages.append({"role": "user", "content": transcript})
            st.session_state.user_query = transcript

            # Detect if user mentioned a location
            if "in " in transcript.lower():
                st.session_state.user_location = transcript.split("in ")[-1].strip()

            # Detect if user mentioned a budget range
            if "between ₹" in transcript.lower():
                budget_part = transcript.split("between ₹")[-1].split(" and ")
                min_budget = int(budget_part[0].strip().replace(",", ""))
                max_budget = int(budget_part[1].strip().replace(",", ""))
                st.session_state.user_budget = (min_budget, max_budget)

            with st.chat_message("user"):
                st.write(transcript)
            os.remove(webm_file_path)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking🤔..."):
            if st.session_state.user_query:
                final_response = get_answer(st.session_state.messages, st.session_state.user_query, st.session_state.user_location, st.session_state.user_budget)
            else:
                final_response = get_answer(st.session_state.messages, st.session_state.user_query, st.session_state.user_location, st.session_state.user_budget)
        with st.spinner("Creating the perfect answer for you...✨"):
            audio_file = text_to_speech(final_response, language=detected_language)
            autoplay_audio(audio_file)
        st.write(final_response)
        st.session_state.messages.append({"role": "assistant", "content": final_response})
        os.remove(audio_file)

# Float the footer container
footer_container.float("bottom: 0rem; right: 10px;")
