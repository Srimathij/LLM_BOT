from gtts import gTTS  

# Define Malay text  
text = "Selamat datang! Bagaimana saya boleh membantu anda hari ini?"  

# Convert text to speech  
tts = gTTS(text=text, lang="ms")  

# Save the output as an audio file  
tts.save("output.mp3")  

# Play the audio (optional, for Windows)  
import os  
os.system("start output.mp3")  
