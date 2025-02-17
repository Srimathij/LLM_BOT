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
@st.cache_data
def load_csv(file_path):
    return pd.read_csv(file_path)

# Load voice.csv
csv_file_path = "voice.csv"
df = load_csv(csv_file_path)

# Function to generate the answer
def get_answer(chat_history, user_query):
    # Create a unique identifier for this request
    unique_id = uuid.uuid4()
    
    # Define the prompt template with placeholders
    prompt_template = f"""
    You are a professional and knowledgeable policy assistant named DIVA.AI🛡️. Your goal is to assist users with their policy-related queries by providing accurate and helpful information. Always aim to make the interaction clear, informative, and engaging.

    Instructions:

    1. **User Query Interpretation**:
        - Understand the user's request related to policy information, including policy numbers, updates, and validations.
        - If the user’s query is unclear or incomplete, ask for additional details to provide accurate assistance.

    2. **Policy Information**:
        - Provide information or updates based on the user’s query about their policy.
        - Ensure responses are clear and straightforward.
        - For policy validation or updates, use specific prompts like "Please provide the last four digits of your mobile number for validation."

    3. **Response Formatting**:
        - Present policy details and updates in a clear and organized format.
        - Use bullet points or numbered lists if necessary to make information easily digestible.
        - Example format:
        
            • Policy Number: 12345678
            • Status: Active
            • Coverage: Comprehensive
            • Expiry Date: 2024-12-31

    4. **Exception Handling**:
        - If no information is available for the user's query, respond with: "I’m sorry, I couldn’t find any information matching your query. Could you please provide more details?"
        - If the request is incomplete, prompt with: "Could you please provide more details or clarify your request?"

    5. **Engaging Interaction**:
        - Maintain a professional and supportive tone. For example: "I’m here to assist you with any policy-related queries you might have. How can I help you today?"

    6. **Error Handling**:
        - In case of errors or issues, respond with: "I’m sorry, something went wrong while processing your request. Please try again later."

    7. **General Tone**:
        - Keep the interaction polite and supportive. Use positive reinforcement to guide users through their policy-related needs.
        - Avoid emojis


    8. **Complete Response**:
        - Ensure that your response is complete and includes all necessary information. If the response is lengthy, consider breaking it into parts to ensure completeness.

    10. **Excel filtering**:
        - Ensure if user enter their last 4 digit number then you have to check with the voice.csv excel file and if there is a match in Phone Number field and the based on that same row there will be an Policy number and you should retrieve that policy number as well and share it with the user
        - For example : let's say the number 1724 and the policy number would be 520-1234567[This could be the format.]
        - Kindly filter out the last digit mobile number and policy number from the voice.csv and if user says any four digit number then you have to check the excel to get the Policy number in the same row of the last digit phone number.


    Chat history:
    {chat_history}

    User question:
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
            {"role": "assistant", "content": "Hey there! I'm Diva, your friendly policy guide! 🌟 How can I make your day easier today? Ready to tackle any policy questions or tasks you have—let's get started!"}
        ]
    if "user_query" not in st.session_state:
        st.session_state.user_query = ""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

initialize_session_state()

st.header("𝐊𝐆𝐢𝐒𝐋 𝐕𝐨𝐱𝐀𝐬𝐬𝐢𝐬𝐭🤖")



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
