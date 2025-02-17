from elevenlabs import ElevenLabs

client = ElevenLabs(
    api_key="297a1180a0284b8a8e7fe9506a4adc82",
)

# Convert text to speech and get audio response
audio_data = client.text_to_speech.convert(
    voice_id="21m00Tcm4TlvDq8ikWAM",
    model_id="eleven_multilingual_v2",
    text="Hello! 你好! Hola! नमस्ते! Bonjour! こんにちは! مرحبا! 안녕하세요! Ciao! Cześć! Привіт! வணக்கம்!",
)

# Save the audio as a file
output_file = "output_audio.mp3"
with open(output_file, "wb") as file:
    file.write(audio_data)

print(f"Audio saved successfully as {output_file}")
