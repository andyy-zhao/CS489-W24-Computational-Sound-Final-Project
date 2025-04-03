import librosa
import numpy as np
import soundfile as sf

def interpolate_freq(idxs: np.ndarray, arr: np.ndarray):
    start = idxs.astype(int)
    frac = (idxs - start)[None, :, None]
    shifted_arr = np.concatenate((arr[:, 1:, :], np.zeros((arr.shape[0], 1, arr.shape[2]))), axis=1)
    return arr[:, start, :] * (1 - frac) + shifted_arr[:, start, :] * frac

def round_interpolate_freq(idxs: np.ndarray, arr: np.ndarray):
    return arr[:, (idxs + 0.5).astype(int), :]

def interpolate_time(idxs: np.ndarray, arr: np.ndarray):
    start = (idxs + 0.5).astype(int)
    frac = (idxs - start)
    shifted_arr = np.concatenate((arr[:, 1:], np.zeros((arr.shape[0], 1))), axis=1)
    return arr[:, start] * (1 - frac) + shifted_arr[:, start] * frac

def round_interpolate_time(idxs: np.ndarray, arr: np.ndarray):
    return arr[:, (idxs + 0.5).astype(int)]  

def pitch_shift():
    y, sr = librosa.load("../data/E4.wav", sr=None, mono=True)
    w_len = 1024 * 4
    X = librosa.stft(y, n_fft=w_len, win_length=w_len)
    num_freqs, num_frames = X.shape
    scaling = 2 ** (2 / 12)
    updated_num_frames = np.floor(num_frames * scaling).astype(int)
    updated_t_frames = np.arange(updated_num_frames)
    original_indices = np.minimum(updated_t_frames / scaling, num_frames - 1)
    magnitude = np.abs(X)
    phases = np.angle(X)
    phase_diffs = phases - np.concatenate((np.zeros((num_freqs, 1)), phases[:, :-1]), axis=1)
    phase_diffs = np.mod(phase_diffs, np.pi * 2)
    shifted_magnitude = interpolate_time(original_indices, magnitude)
    shifted_phase_diffs = interpolate_time(original_indices, phase_diffs)
    unshifted_phases = round_interpolate_time(original_indices, phases)
    shifted_phases = np.zeros((num_freqs, updated_num_frames))
    shifted_phases[:, 0] = shifted_phase_diffs[:, 0]

    for t in range(1, updated_num_frames):
        time_phases = shifted_phases[:, t - 1] + shifted_phase_diffs[:, t]
        freq_phases = unshifted_phases[:, t]
        transient = (shifted_magnitude[:, t] - shifted_magnitude[:, t - 1]) / (shifted_magnitude[:, t] + shifted_magnitude[:, t - 1])
        transient[transient < 0.5] = 0
        transient[transient >= 0.5] = 1
        shifted_phases[:, t] = np.mod(freq_phases * transient + time_phases * (1 - transient), np.pi * 2)

    synth_stft = shifted_magnitude * np.exp(shifted_phases * 1j)
    new_waveform = librosa.istft(synth_stft, n_fft=w_len, window="hann")
    sf.write('../data/pitch_shifted_output.wav', new_waveform, int(sr * scaling))

pitch_shift()