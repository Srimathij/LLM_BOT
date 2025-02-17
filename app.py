
import streamlit as st
import os
import uuid
import base64
import json
import speech_recognition as sr
from gtts import gTTS
from audio_recorder_streamlit import audio_recorder
from streamlit_float import *
from dotenv import load_dotenv
from groq import Groq
import librosa
import soundfile as sf
from langdetect import detect

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq client for Whisper
client = Groq()

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
    # Create a unique identifier for this request
    unique_id = uuid.uuid4()
    
    # If location and budget are provided, fetch matching rooms from JSON
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
    Hey! I'm Booking.AI, your hotel booking assistant. I'll guide you through booking a room step by step. No need to repeat greetings—let's just focus on making your experience smooth and efficient.

    ### Booking Flow Instructions:

    1. Location Request:
        - I'll first ask for the city where you want to stay.
        - Once you provide the city, I’ll ask for your check-in and check-out dates.

    2. Date Inquiry:
        - After I get the city, I’ll ask for the check-in date first, followed by the check-out date.
        - If both dates are provided, I’ll inquire about your budget per night.

    3. Budget Inquiry:
        - Once I know the dates, I'll ask about your budget range to ensure the best room options for you.
        - If you provide all the information, I’ll proceed to confirm the details in a single message, ensuring all steps are captured without redundancy.

    4. Confirmation (after collecting all information):
        - After gathering the city, check-in and check-out dates, and budget, I’ll confirm all the information for accuracy.
        - This avoids repetitive prompts unless further clarification is needed.
        - Example confirmation: "You're looking for a room in {{city}} from {{check_in_date}} to {{check_out_date}} with a budget of {{budget}} per night. Is that correct?"

    5. Room Search and Response:
        - After confirmation, I’ll provide you with available room options including:
            1. Room name, location, and price per night
            2. Key amenities like free breakfast, Wi-Fi, or room service
            3. Ratings and discounts

    6. Booking Confirmation:
        - Once you choose a room, I’ll confirm the booking and provide you with a reference number.

    7. Error Handling and Clarification:
        - If I need additional details or clarification, I’ll ask politely without repeating previous requests.
        - The conversation will flow smoothly without getting stuck on earlier steps.

    ### Example conversation:

    You: "I’d like to book a hotel room in Coimbatore."
        
    Me: "Got it! Which dates would you like to check in and check out?"

    You: "October 5th to October 8th."

    Me: "Great! And what’s your budget per night?"

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
    # Debugging: Print the prompt template to check formatting
    print("Prompt Template:\n", prompt_template)
    
    # Use Groq's chat completion endpoint with the LLaMA model
    try:
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "system", "content": prompt_template}],
            temperature=0.7,
            max_tokens=2000,
            top_p=1,
            stream=False,
        )
        
        # Debugging: Print the raw API response
        print("API Response:\n", completion)
        
        # Retrieve the response
        response_text = completion.choices[0].message.content.strip()
        return response_text
    
    except Exception as e:
        print("Error occurred:", e)
        return "An error occurred while processing your request."

# Updated speech-to-text function using Whisper model
# Updated speech-to-text function using Whisper model
def speech_to_text(audio_path):
    processed_audio_path = preprocess_audio(audio_path)
    
    with open(processed_audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3",
            response_format="verbose_json",
        )
    
    # Access the transcription text directly
    recognized_text = transcription.text if hasattr(transcription, 'text') else ""

    if not recognized_text:
        print("No text recognized.")
        return ""
    
    # Detect the language and log the detection results
    try:
        detected_language = detect(recognized_text)
        print(f"Detected language: {detected_language}")
    except Exception as e:
        print(f"Language detection failed: {e}")
        detected_language = 'unknown'

    print(f"User said: {recognized_text}")
    
    # Allow English or fallback to unknown languages
    if detected_language == 'en' or detected_language == 'unknown':
        return recognized_text
    else:
        print("Non-English text detected. Returning empty string.")
        return ""


# Example usage of the updated speech_to_text function
audio_path = "Adver_converted.wav"
recognized_text = speech_to_text(audio_path)
print(f"Recognized text: {recognized_text}")


# Function to convert text to speech
def text_to_speech(input_text):
    # Use gTTS to convert text to speech and save the audio
    tts = gTTS(text=input_text, lang='en')
    wav_file_path = "temp_audio_play.mp3"
    tts.save(wav_file_path)
    return wav_file_path

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

# Float feature initialization
float_init()

# Initialize session state
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
        st.session_state.user_budget = (0, float("inf"))  # Default budget range

initialize_session_state()

st.header("𝑸𝒖𝒊𝒄𝒌𝑩𝒐𝒐𝒌.𝙰𝙸✨")

# Create footer container for the microphone
footer_container = st.container()
with footer_container:
    audio_bytes = audio_recorder(text=None, icon_size="3x", sample_rate=16000)

# Display messages
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




