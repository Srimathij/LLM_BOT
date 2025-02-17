import streamlit as st
import os
import uuid
import base64
import pandas as pd
from dotenv import load_dotenv
import speech_recognition as sr
from gtts import gTTS
from audio_recorder_streamlit import audio_recorder
from streamlit_float import *
from groq import Groq

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq client for LLaMA
client = Groq()

# Load the CSV file (voice.csv) with policy and phone numbers
# @st.cache_data
# def load_csv(file_path):
#     return pd.read_csv(file_path)

# # Load voice.csv
# csv_file_path = "voice.csv"
# df = load_csv(csv_file_path)

# Function to generate the answer
def get_answer(chat_history, user_query):
    # Create a unique identifier for this request
    unique_id = uuid.uuid4()
    
   # Define the prompt template with placeholders
    prompt_template = f"""
    Hey there!  I'm OYO.AI, your friendly hotel booking assistant for OYO hotels.  Happy to help you find the perfect room! 

    **Instructions:**

    1. **User Query Interpretation:**
        - Listen carefully to what you'd like to book, including the city, dates, and budget.
        - If anything seems unclear, I'll politely ask for more details to complete your request.

    2. **Room Search and Information:**
        - To get started, tell me which city you'd like to stay in.

    3. **Response Formatting:**
        - I'll present clear and concise options for available rooms, including details like:
            • Room name, location, and price per night
            • Amenities like free breakfast, Wi-Fi, or room service

    4. **Handling Uncertainty:**
        - If no rooms match your criteria, I'll suggest adjusting your search or offer alternative locations.
        - If I need more information, I'll ask politely for specifics like budget or preferred area.

    5. **Booking Confirmation:**
        - Once you've chosen a room, I'll confirm the booking and provide a reference number.

    6. **Error Handling:**
        - If something goes wrong, I'll apologize and suggest trying again later.

    7. **General Tone:**
        - I'll maintain a friendly, professional, and helpful tone throughout our conversation.
        - No more repetitive greetings! I'll use natural conversation flow to keep things engaging.

    8. **Complete Response:**
        - My responses will be comprehensive, providing all the necessary details to help you make informed decisions.

    **Example conversation:**

    **You:** "I'd like to book a hotel room in Coimbatore."

    **Me:** "Great choice! When would you like to check in?"

    **You:** "October 5th."

    **Me:** "And what's your check-out date?"

    **You:** "October 8th."

    **Me:** "Okay, and what's your budget per night?"

    **You:** "Between ₹2,000 and ₹4,000."

    **Me:** "Here are some options that might work for you:

        • Room 1: Collection O Darshans Abode 47, Siddhapudur (₹2,500/night, Free Breakfast, Free Wi-Fi)
        • Room 2: Townhouse 1123 Four Season 369 B (₹3,000/night, 24/7 Room Service, Free Parking)"

    **Chat history:**
    {chat_history}

    **Your question:**
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
            stream=False,  # Set to False if you want the entire response at once
        )
        
        # Debugging: Print the raw API response
        print("API Response:\n", completion)
        
        # Retrieve the response
        response_text = completion.choices[0].message.content.strip()
        return response_text
    
    except Exception as e:
        print("Error occurred:", e)
        return "An error occurred while processing your request."

def speech_to_text(audio_file):
    recognizer = sr.Recognizer()
    
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
    
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "Sorry, I could not understand the audio."
    except sr.RequestError as e:
        return f"Error with the speech recognition service: {e}"

def text_to_speech(input_text):
    # Use gTTS to convert text to speech and save the audio
    tts = gTTS(text=input_text, lang='en')
    wav_file_path = "temp_audio_play.mp3"
    tts.save(wav_file_path)
    return wav_file_path

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

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hey there! 😊 Welcome to OYO! I'm here to help you find the perfect place to stay. Could you please share your name with me so we can get started?"}
        ]
    if "user_query" not in st.session_state:
        st.session_state.user_query = ""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

initialize_session_state()

st.header("𝑶𝒀𝑶 𝑸𝒖𝒊𝒄𝒌𝑩𝒐𝒐𝒌✨")



# Create footer container for the microphone
footer_container = st.container()
with footer_container:
    audio_bytes = audio_recorder(text=None, icon_size="3x", sample_rate=16000)

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if audio_bytes:
    with st.spinner("Processing your request..📜"):
        webm_file_path = "temp_audio.mp3"
        with open(webm_file_path, "wb") as f:
            f.write(audio_bytes)
        transcript = speech_to_text(webm_file_path)
        if transcript:
            st.session_state.messages.append({"role": "user", "content": transcript})
            st.session_state.user_query = transcript
            with st.chat_message("user"):
                st.write(transcript)
            os.remove(webm_file_path)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking🤔..."):
            if "show me the data" in st.session_state.user_query.lower():
                # Display the CSV data
                st.write("Here is the data from the CSV file:")
                st.dataframe(df)
                final_response = "Here is the data from the CSV file:"
            else:
                if st.session_state.user_query:
                    final_response = get_answer(st.session_state.messages, st.session_state.user_query)
                else:
                    final_response = get_answer(st.session_state.messages, st.session_state.user_query)
        with st.spinner("Generating response...🤔"):
            audio_file = text_to_speech(final_response)
            autoplay_audio(audio_file)
        st.write(final_response)
        st.session_state.messages.append({"role": "assistant", "content": final_response})
        os.remove(audio_file)

# Float the footer container
footer_container.float("bottom: 0rem;right: 10px;")
