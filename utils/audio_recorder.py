import pyaudio
import wave

# Audio Settings
FORMAT = pyaudio.paInt16  # 16-bit audio format
CHANNELS = 1              # Mono
RATE = 44100              # 44.1 kHz sample rate
CHUNK = 1024              # Buffer size
DURATION = 5              # Recording duration in seconds
OUTPUT_FILE = "output.wav"

# Initialize PyAudio
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)

print("Recording...")
frames = []

# Record in real-time
for _ in range(0, int(RATE / CHUNK * DURATION)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Recording complete.")

# Stop and close the stream
stream.stop_stream()
stream.close()
audio.terminate()

# Save as a .wav file
with wave.open(OUTPUT_FILE, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

print(f"Audio saved to {OUTPUT_FILE}")
