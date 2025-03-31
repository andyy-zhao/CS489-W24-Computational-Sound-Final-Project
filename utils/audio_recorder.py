import pyaudio
import wave

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
DURATION = 5
OUTPUT_FILE = "output.wav"

audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)

print("Recording...")
frames = []

for _ in range(0, int(RATE / CHUNK * DURATION)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Recording complete.")

stream.stop_stream()
stream.close()
audio.terminate()

with wave.open(OUTPUT_FILE, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

print(f"Audio saved to {OUTPUT_FILE}")
