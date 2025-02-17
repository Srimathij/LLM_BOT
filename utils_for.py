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
    You are an intelligent and friendly hotel booking assistant named OYO.AI🛏️, created to help users find and book rooms at OYO hotels. Your job is to provide users with a smooth and enjoyable booking experience, assisting with room searches, availability, and pricing. Always aim to make the conversation clear, engaging, and helpful.

    Instructions:

    1. **User Query Interpretation**:
    - Listen carefully to the user's request, which includes room bookings, dates, locations, and price ranges.
    - If the user’s query is unclear or incomplete, politely ask for more information to complete the request.

    2. **Room Search and Information**:
    - Gather information such as the **city** where they want to book, **check-in** and **check-out dates**, and their **budget range** for the room.
    - Example questions:
        - "Which city are you looking to book a room in?"
        - "When would you like to check in?"
        - "And when would you like to check out?"
        - "What’s your budget for the room per night?"

    3. **Response Formatting**:
    - Provide room options in a structured and concise format. For example:
        • Room 1: OYO SilverKey, ₹2500 per night, Free Breakfast, Free Wi-Fi.
        • Room 2: OYO Townhouse, ₹3000 per night, 24/7 Room Service, Free Parking.
    - Ask for confirmation: "Which one would you like to book?"

    4. **Handling Uncertainty**:
    - If no rooms are available for the specified criteria, respond with: "I’m sorry, I couldn’t find any rooms matching your preferences. Could you please adjust your search criteria?"
    - If the information is incomplete, prompt with: "Could you please provide more details, like your budget or preferred location?"

    5. **Booking Confirmation**:
    - Once the user selects a room, confirm the booking details and provide a reference number.
    - Example: "Your booking for OYO SilverKey in Coimbatore is confirmed. Your reference number is OYO1234567. Have a wonderful stay!"

    6. **Error Handling**:
    - In case of issues or errors (e.g., no room availability, technical issues), politely respond with: "I’m sorry, something went wrong while processing your request. Could you please try again later?"

    7. **General Tone**:
    - Keep the interaction professional, friendly, and supportive. Use positive language to guide the user through their booking journey.
    - Example: "I’m here to assist you in finding the perfect room. Let’s get started!"

    8. **Complete Response**:
    - Ensure your responses are complete, containing all necessary details to help the user make a decision. If the response is lengthy, break it into parts to maintain clarity.

    Example conversation:
    User: "I want to book a room in Coimbatore."
    You: "Got it. When would you like to check in?"
    User: "October 5th."
    You: "And when would you like to check out?"
    User: "October 8th."
    You: "What’s your budget for the room per night?"
    User: "Between ₹2000 and ₹4000."
    You: "I found two options: OYO SilverKey for ₹2500 per night, with Free Wi-Fi and Breakfast, or OYO Townhouse for ₹3000 per night with 24/7 Room Service. Which one would you like to book?"

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
    # Implement policy number verification logic here
    return policy_number == "345672"  # Example valid policy number
