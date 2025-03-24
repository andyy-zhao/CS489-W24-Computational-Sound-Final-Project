# import pyaudio
# import numpy as np
# import soundfile as sf
# import os
# from utils.pitch_data import NOTE_FREQUENCIES, get_closest_note
# from models.pitch_shifter import pitch_shift
# # from models.real_time_pitch_detector import detect_pitch_from_audio

# # Audio parameters
# fs = 44100
# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# CHUNK = 1024 * 4
# THRESHOLD = 700

# def get_semitone_shift(current_freq, target_freq):
#     """Calculate the number of semitones between two frequencies"""
#     return 12 * np.log2(target_freq / current_freq)

# def autotune():
#     p = pyaudio.PyAudio()
    
#     # Open input stream
#     stream = p.open(
#         format=FORMAT,
#         channels=CHANNELS,
#         rate=fs,
#         input=True,
#         frames_per_buffer=CHUNK
#     )

#     print('Recording started... Press Ctrl+C to stop')
    
#     # Initialize array to store processed chunks
#     processed_chunks = []
    
#     try:
#         while True:
#             # Read audio data
#             data = stream.read(CHUNK)
#             audio_data = np.frombuffer(data, dtype=np.int16)
            
#             # Detect pitch and get target note
#             current_note, max_freq = detect_pitch_from_audio(audio_data)
#             if current_note is not None:
#                 target_freq = NOTE_FREQUENCIES[current_note]
#                 semitones = get_semitone_shift(max_freq, target_freq)
#                 pitch_shift_factor = 2 ** (semitones / 12)
                
#                 # # Convert chunk to float for processing
#                 # chunk_float = audio_data.astype(np.float32) / 32768.0
                
#                 # # Apply pitch shift to this chunk
#                 # shifted_chunk = pitch_shift(chunk_float, pitch_shift_factor, CHUNK)
                
#                 # # Convert back to 16-bit and store
#                 # shifted_chunk = np.int16(shifted_chunk * 32767)
#                 # processed_chunks.append(shifted_chunk)
            
#     except KeyboardInterrupt:
#         print('\nRecording stopped. Processing audio...')
        
#     finally:
#         stream.stop_stream()
#         stream.close()
#         p.terminate()
    
#     # Concatenate all processed chunks
#     autotuned_audio = np.concatenate(processed_chunks)
    
#     # Create data directory if it doesn't exist
#     os.makedirs('../data', exist_ok=True)
    
#     # Save the autotuned audio
#     sf.write('./data/autotuned.wav', autotuned_audio, fs)
#     print('Autotuned audio saved to data/autotuned.wav')

# if __name__ == "__main__":
#     autotune()
