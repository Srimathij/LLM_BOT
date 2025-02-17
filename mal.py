import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the model and tokenizer
model = ParlerTTSForConditionalGeneration.from_pretrained("mesolitica/malay-parler-tts-mini-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("mesolitica/malay-parler-tts-mini-v1")

# List of speakers
speakers = [
    'Yasmin',
   
]

# The text prompt you want to convert to speech
prompt = 'Husein zolkepli sangat comel dan kacak suka makan cendol'

# Generate speech for each speaker
for s in speakers:
    description = f"{s}'s voice, delivers a slightly exprRessive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."

    # Tokenize the description and prompt
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Generate the audio
    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    
    # Move the audio array to CPU and save it as a .mp3 file
    audio_arr = generation.cpu().numpy().squeeze()
    sf.write(f'{s}.mp3', audio_arr, 44100)

print("Audio files generated for all speakers.")

