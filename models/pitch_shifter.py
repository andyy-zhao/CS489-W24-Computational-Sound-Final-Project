import numpy as np
import librosa

def interpolate_time(idxs: np.ndarray, arr: np.ndarray):
    start = (idxs + 0.5).astype(int)  # Start indices for interpolation
    frac = (idxs - start)  # Fractional part for interpolation
    shifted_arr = np.concatenate((arr[:, 1:], np.zeros((arr.shape[0], 1))), axis=1)  # Shift arr by 1 along time axis
    return arr[:, start] * (1 - frac) + shifted_arr[:, start] * frac  # Interpolation along time axis

def round_interpolate_time(idxs: np.ndarray, arr: np.ndarray):
    return arr[:, (idxs + 0.5).astype(int)]

def pitch_shift(audio_data: np.ndarray, pitch_shift: float, n_fft: int) -> np.ndarray:
    """
    Apply pitch shifting to audio data using phase vocoding.
    
    Args:
        audio_data: Input audio data (float32, range [-1, 1])
        pitch_shift: Pitch shift factor (e.g., 2.0 for one octave up)
        n_fft: FFT size
    
    Returns:
        Pitch-shifted audio data
    """
    # Compute STFT
    X = librosa.stft(audio_data, n_fft=n_fft, win_length=n_fft)
    num_freqs, num_frames = X.shape
    
    # Get magnitude and phase
    magnitude = np.abs(X)
    phases = np.angle(X)
    
    # Calculate phase differences
    phase_diffs = phases - np.concatenate((np.zeros((num_freqs, 1)), phases[:, :-1]), axis=1)
    phase_diffs = np.mod(phase_diffs, np.pi * 2)
    
    # Calculate new number of frames after pitch shift
    updated_num_frames = np.floor(num_frames * pitch_shift).astype(int)
    updated_t_frames = np.arange(updated_num_frames)
    
    # Calculate original indices for interpolation
    original_indices = np.minimum(updated_t_frames / pitch_shift, num_frames - 1)
    
    # Interpolate magnitude and phase differences
    shifted_magnitude = interpolate_time(original_indices, magnitude)
    shifted_phase_diffs = interpolate_time(original_indices, phase_diffs)
    unshifted_phases = round_interpolate_time(original_indices, phases)
    
    # Initialize shifted phases array
    shifted_phases = np.zeros((num_freqs, updated_num_frames))
    shifted_phases[:, 0] = shifted_phase_diffs[:, 0]
    
    # Process each frame
    for t in range(1, updated_num_frames):
        # Accumulate phase from previous frame and current phase difference
        time_phases = shifted_phases[:, t - 1] + shifted_phase_diffs[:, t]
        freq_phases = unshifted_phases[:, t]
        
        # Compute transient factor for smooth transitions
        transient = (shifted_magnitude[:, t] - shifted_magnitude[:, t - 1]) / (shifted_magnitude[:, t] + shifted_magnitude[:, t - 1])
        transient[transient < 0.5] = 0
        transient[transient >= 0.5] = 1
        
        # Blend phases based on transient detection
        shifted_phases[:, t] = np.mod(freq_phases * transient + time_phases * (1 - transient), np.pi * 2)
    
    # Reconstruct signal
    shifted_stft = shifted_magnitude * np.exp(1j * shifted_phases)
    shifted_audio = librosa.istft(shifted_stft, n_fft=n_fft, window="hann")
    
    return shifted_audio 