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
from TTS.api import TTS  # Import the TTS library

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq client for Whisper
client = Groq()

# Initialize TTS model
tts_model_name = "tts_models/en/ljspeech/glow-tts"  # English LJSpeech model
tts = TTS(model_name=tts_model_name)

# Preprocess audio function
def preprocess_audio(audio_path):
    # Load the audio file using librosa
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Save preprocessed audio to a temporary file
    temp_audio_path = f"temp_{uuid.uuid4()}.wav"
    sf.write(temp_audio_path, audio, sr)  # Save using soundfile
    
    return temp_audio_path

# Load the JSON file (rooms.json) with hotel data
@st.cache_data
def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Load rooms.json
json_file_path = "rooms.json"
rooms_data = load_json(json_file_path)

# Function to search rooms based on location, budget, and user preferences
def search_rooms(location, budget_range):
    min_budget, max_budget = budget_range
    results = [room for room in rooms_data if room['Location'].lower() == location.lower() and min_budget <= room['Budget'] <= max_budget]
    return results

# Function to generate the answer
def get_answer(chat_history, user_query, user_location, user_budget):
    unique_id = uuid.uuid4()

    room_options = ""
    if user_location and user_budget:
        matching_rooms = search_rooms(user_location, user_budget)
        if matching_rooms:
            rooms_list = "\n".join([f" {room['Hotel_name']} in {room['Location']} (₹{room['Budget']}/night, {room['Amenities']}, Rating: {room['Rating']}⭐, Discount: {room['Discount']})"
                                    for room in matching_rooms])
            room_options = f"Here are some hotel options for {user_location} within your budget:\n\n{rooms_list}"
        else:
            room_options = f"Sorry, we couldn't find any hotel options in {user_location} within your budget."

    prompt_template = f"""
    Hey! I'm OYO.AI, your hotel booking assistant. I'll help you book a room quickly and efficiently. Let's focus on making this process smooth, without repeating already confirmed details. Please make sure to respond completely and provide all necessary information in your answers.

    ### Booking Flow Instructions:

    1. Location Request:
        - I will first ask for the city where you'd like to stay.
        - After that, I’ll ask for your check-in and check-out dates, and your budget per night.
        - I won't ask for the location again once it's provided.

    2. Date and Budget Inquiry:
        - Once I have the city, I'll ask for your check-in and check-out dates.
        - After that, I'll ask for your budget range per night.
        - No need to repeat the dates or location unless you want to change them.

    3. Confirmation:
        - I’ll confirm your details after gathering the city, check-in and check-out dates, and budget.
        - Please confirm that I mention all key details correctly, including city, dates, and budget. For example: 
        "You're looking for a room in {{city}} from {{check_in_date}} to {{check_out_date}} with a budget of {{budget}} per night. Is that correct?"

    4. Room Search and Response:
        - Once you confirm the details, I’ll show you room options that fit your preferences, including:
            1. Hotel name, location, and price per night
            2. Key amenities like free breakfast, Wi-Fi, or room service
            3. Ratings and discounts
        - Please provide at least 2 room options to choose from.

    5. Booking Confirmation:
        - After you choose a room, I’ll confirm the booking and provide you with a reference number. The process is quick and easy!

    7. Error Handling and Clarification:
        - If you need additional details or clarification, I’ll ask politely without repeating previous requests.
        - The conversation will flow smoothly without getting stuck on earlier steps.

    ### Example conversation:

    You: "I’d like to book a hotel room in Coimbatore."
        
    Me: "Got it! Which dates would you like to check in and check out?"

    You: "October 5th to October 8th."

    Me: "Great! What’s your budget per night?"

    You: "Between ₹2,000 and ₹4,000."

    Me: "Just to confirm: you're looking for a room in Coimbatore from October 5th to October 8th with a budget between ₹2,000 and ₹4,000 per night. Is that correct?"

    You: "{room_options}"

    You: "I’ll go with Room 1."

    Me: "Your booking is confirmed! Here’s your reference number: 12782ABF. Have a great stay!"

    ### Chat History:
    {chat_history}

    ### Your question:
    {user_query}
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
        return ""
    
    try:
        detected_language = detect(recognized_text)
        print(f"Detected language: {detected_language}")
    except Exception as e:
        print(f"Language detection failed: {e}")
        detected_language = 'unknown'

    if detected_language == 'en' or detected_language == 'unknown':
        return recognized_text
    else:
        print("Non-English text detected. Returning empty string.")
        return ""

# Function to convert text to speech using TTS
def text_to_speech(input_text, speed=0.5):
    wav_file_path = "temp_audio_play.wav"
    # Adjust the `speed` parameter to make the voice faster
    tts.tts_to_file(text=input_text, file_path=wav_file_path, speed=speed, sample_rate=16000, gpu=True)
    return wav_file_path

# Function to autoplay audio in Streamlit
def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    md = f"""
    <audio autoplay>
    <source src="data:audio/wav;base64,{b64}" type="audio/wav">
    </audio>
    """
    st.markdown(md, unsafe_allow_html=True)

# Float feature initialization
float_init()

# Initialize session state
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hey there! 😊 Welcome to OYO! I'm here to help you find the perfect place to stay. Could you please share your name with me so we can get started?"}
        ]
    if "user_query" not in st.session_state:
        st.session_state.user_query = ""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "user_location" not in st.session_state:
        st.session_state.user_location = None
    if "user_budget" not in st.session_state:
        st.session_state.user_budget = (0, float("inf"))  # Default budget range
    if "voice_stopped" not in st.session_state:
        st.session_state.voice_stopped = False  # New state to track voice stopping

initialize_session_state()

st.header("𝑶𝒀𝑶 𝑸𝒖𝒊𝒄𝒌𝑩𝒐𝒐𝒌✨")

# Create footer container for the microphone and Stop button
# Create footer container for the microphone and Stop button
# Create footer container for the microphone and Stop/Resume buttons
footer_container = st.container()
with footer_container:
    col1, col2, col3 = st.columns([2, 1, 1])
    
    # Microphone recording logic
    with col1:
        if not st.session_state.voice_stopped:  # Allow recording only if voice is not stopped
            audio_bytes = audio_recorder(text=None, icon_size="2x", sample_rate=16000)
        else:
            st.write("----")
    
    # Stop voice response button
    with col2:
        stop_button = st.button("Stop speaking 🔇")
    
    # Resume voice response button
    with col3:
        if st.session_state.voice_stopped:
            resume_button = st.button("Resume🎤")

# Handle Stop Voice Response button click
if stop_button:
    st.session_state.voice_stopped = True  # Set voice stopped to True

# Handle Resume Voice Response button click
if 'resume_button' in locals() and resume_button:
    st.session_state.voice_stopped = False  # Reset voice stopped to False to enable mic again

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Process audio if recording is allowed
if audio_bytes and not st.session_state.voice_stopped:
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

# Generate assistant's response if required
if st.session_state.messages[-1]["role"] != "assistant" and not st.session_state.voice_stopped:
    with st.chat_message("assistant"):
        with st.spinner("Thinking🤔..."):
            final_response = get_answer(st.session_state.messages, st.session_state.user_query, st.session_state.user_location, st.session_state.user_budget)
        with st.spinner("Creating the perfect answer for you...✨"):
            audio_file = text_to_speech(final_response)
            autoplay_audio(audio_file)
        st.write(final_response)
        st.session_state.messages.append({"role": "assistant", "content": final_response})
        os.remove(audio_file)

# Float the footer container
footer_container.float("bottom: 0rem;right: 10px;")
