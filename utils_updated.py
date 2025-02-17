import os
import time
from dotenv import load_dotenv
import base64
import requests
import speech_recognition as sr
from gtts import gTTS  # Import gTTS
import streamlit as st
import uuid
from groq import Groq

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq()

def get_answer(chat_history, user_query):
    # Create a unique identifier for this request
    unique_id = uuid.uuid4()
    
    # Define the prompt template with placeholders
    prompt_template = f"""
    You are a professional and knowledgeable policy assistant named PolicyHelper.AI🛡️. Your goal is to assist users with their policy-related queries by providing accurate and helpful information. Always aim to make the interaction clear, informative, and engaging.

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
        - Avoid using emojis
t
    8. **Complete Response**:
        - Ensure that your response is complete and includes all necessary information. If the response is lengthy, consider breaking it into parts to ensure completeness.

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
        text = recognizer.recognize_google(audio_data)  # Use Google's speech recognition
        return text
    except sr.UnknownValueError:
        return "Sorry, I could not understand the audio."
    except sr.RequestError as e:
        return f"Error with the speech recognition service: {e}"

def text_to_speech(input_text):
    # Synthesize speech from text using gTTS
    tts = gTTS(text=input_text, lang='en', slow=False)
    wav_file_path = "temp_audio_play.mp3"  # Saving the file as mp3
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

def verify_policy_number(policy_number):
    return policy_number == "345672"  
