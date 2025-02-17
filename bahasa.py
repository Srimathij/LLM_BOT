import streamlit as st
import os
import uuid
import base64
import json
import speech_recognition as sr
from audio_recorder_streamlit import audio_recorder
from streamlit_float import *
from dotenv import load_dotenv
from groq import Groq
import librosa
import soundfile as sf
from langdetect import detect
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
 
# Initialize Groq client for Whisper
client = Groq()

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the ParlerTTS model and tokenizer for Malay
model = ParlerTTSForConditionalGeneration.from_pretrained("mesolitica/malay-parler-tts-mini-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("mesolitica/malay-parler-tts-mini-v1")

# Function to generate speech in Malay
def text_to_speech_malay(input_text, speaker='Yasmin'):
    description = f"{speaker}'s voice, delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."
    
    # Tokenize the description and prompt
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

    # Generate the audio
    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)

    # Move the audio array to CPU and save it as a .mp3 file
    audio_arr = generation.cpu().numpy().squeeze()
    wav_file_path = f"{uuid.uuid4()}.mp3"
    sf.write(wav_file_path, audio_arr, 44100)

    return wav_file_path

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

def get_answer(chat_history, user_query, user_location, user_budget):
    unique_id = uuid.uuid4()

    room_options = ""
    if user_location and user_budget:
        matching_rooms = search_rooms(user_location, user_budget)
        if matching_rooms:
            rooms_list = "\n".join([f" {room['Hotel_name']} in {room['Location']} (\u20b9{room['Budget']}/night, {room['Amenities']}, Rating: {room['Rating']}\u2b50, Discount: {room['Discount']})"
                                    for room in matching_rooms])
            room_options = f"Here are some hotel options for {user_location} within your budget:\n\n{rooms_list}"
        else:
            room_options = f"Sorry, we couldn't find any hotel options in {user_location} within your budget."

    prompt_template = f"""
    ### Chat History:
    {chat_history}

    ### Your question:
    {user_query}

    {room_options}
    """
    
    try:
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "system", "content": prompt_template}],
            temperature=0.7,
            max_tokens=2000,
            top_p=1,
            stream=False,
        )

        response_text = completion.choices[0].message.content.strip()
        return response_text

    except Exception as e:
        print("Error occurred:", e)
        return "An error occurred while processing your request."

# Updated speech-to-text function using Whisper model
# Updated speech-to-text function to handle English and Malay
def speech_to_text(audio_path):
    processed_audio_path = preprocess_audio(audio_path)

    with open(processed_audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3",
            response_format="verbose_json",
        )

    recognized_text = transcription.text if hasattr(transcription, 'text') else ""

    if not recognized_text:
        print("No text recognized.")
        return "", "unknown"

    try:
        detected_language = detect(recognized_text)
        print(f"Detected language: {detected_language}")
    except Exception as e:
        print(f"Language detection failed: {e}")
        detected_language = 'unknown'

    print(f"User said: {recognized_text}")

    return recognized_text, detected_language

float_init()

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hey there! 😊 Welcome to Booking.AI! I'm here to help you find the perfect place to stay. Could you please share your name with me so we can get started?"}
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

# Center the header using Streamlit's HTML rendering
st.markdown("""
    <style>
    .header {
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        margin-top: 10px;
    }
    .mic-container {
        position: fixed;
        bottom: 1rem;
        right: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='header'>VoiceBot.AI✨</div>", unsafe_allow_html=True)

footer_container = st.container()
with footer_container:
    audio_bytes = audio_recorder(
    text="",
    recording_color="#e8b62c",
    neutral_color="#6aa36f",
    icon_name="microphone-lines",
    icon_size="2x",
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if audio_bytes:
    with st.spinner("Hang on! I’m finding the best options for you...⏳"):
        webm_file_path = "temp_audio.mp3"
        with open(webm_file_path, "wb") as f:
            f.write(audio_bytes)
        transcript = speech_to_text(webm_file_path)
        if transcript:
            st.session_state.messages.append({"role": "user", "content": transcript})
            st.session_state.user_query = transcript
            
            # Detect if user mentioned a location (you can improve this detection logic)
            if "in " in transcript.lower():
                st.session_state.user_location = transcript.split("in ")[-1].strip()
            
            # Detect if user mentioned a budget range (you can improve this logic)
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
            audio_file = text_to_speech(final_response)
            autoplay_audio(audio_file)
        st.write(final_response)
        st.session_state.messages.append({"role": "assistant", "content": final_response})
        os.remove(audio_file)


# Float the footer container
footer_container.float("bottom: 0rem;right: 10px;")

