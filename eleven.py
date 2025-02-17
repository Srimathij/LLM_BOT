from elevenlabs import ElevenLabs

client = ElevenLabs(
    api_key="sk_edcb9fcb8509b77082fb28d73a90fdc57fb3cc942adc1349",
)

# Convert text to speech and get audio response (generator)
audio_generator = client.text_to_speech.convert(
    voice_id="21m00Tcm4TlvDq8ikWAM",
    model_id="eleven_multilingual_v2",
    text="Hello! 你好! Hola! नमस्ते! Bonjour! こんにちは! مرحبا! 안녕하세요! Ciao! Cześć! Привіт! வணக்கம்!",
)

# Save the audio as a file
output_file = "output_audio.mp3"
with open(output_file, "wb") as file:
    for chunk in audio_generator:
        file.write(chunk)

print(f"Audio saved successfully as {output_file}")
