import os
import tempfile
import sounddevice as sd
import scipy.io.wavfile as wav
from faster_whisper import WhisperModel
import ollama
import time
import keyboard
import numpy as np

# Load Whisper model on CPU
print("🔁 Loading Whisper model...")
whisper = WhisperModel("base", compute_type="int8", device="cpu")

def record_audio(fs=16000):
    print("🎙️ Press [Enter] to start listening.")
    while True:
        if keyboard.is_pressed("enter"):
            print("🎧 Listening... Press [Enter] again to stop.")
            break
        time.sleep(0.1)

    recording = []

    while True:
        chunk = sd.rec(int(1 * fs), samplerate=fs, channels=1)
        sd.wait()
        recording.append(chunk)

        if keyboard.is_pressed("enter"):
            print("🛑 Stopped listening.")
            # wait for key to be released before next loop
            while keyboard.is_pressed("enter"):
                time.sleep(0.1)
            break
        time.sleep(0.1)

    return fs, np.concatenate(recording, axis=0)

def save_temp_wav(fs, audio_data):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        wav.write(tmp.name, fs, audio_data)
        return tmp.name

def transcribe(audio_path):
    segments, _ = whisper.transcribe(audio_path)
    transcription = " ".join([seg.text for seg in segments])
    return transcription.strip()

def chat_with_llama(prompt):
    response = ollama.chat(
        model='llama3',
        messages=[
            {
                "role": "system",
                "content": (
                    "You're a helpful and witty social sidekick. Help the user fit in by offering natural, friendly, and "
                    "sometimes funny responses in social situations."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response['message']['content']

if __name__ == "__main__":
    try:
        while True:
            fs, audio_data = record_audio()
            temp_path = save_temp_wav(fs, audio_data)
            text = transcribe(temp_path)
            os.remove(temp_path)

            if text:
                print(f"\n👂 Heard: {text}")
                print("🤖 Generating reply...")
                reply = chat_with_llama(text)
                print(f"\n💬 Sidekick: {reply}\n")
            else:
                print("❌ Didn't catch anything.\n")

            print("💤 Waiting for you to press [Enter] to start listening again...\n")

    except KeyboardInterrupt:
        print("\n👋 Exiting... Bye!")
