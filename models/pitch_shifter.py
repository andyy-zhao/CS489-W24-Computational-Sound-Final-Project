import librosa
import numpy as np
import soundfile as sf

class PitchShifter:
    def __init__(self):
        self.sr = None
        self.w_len = 1024 * 4

    def interpolate_time(self, idxs: np.ndarray, arr: np.ndarray):
        start = (idxs + 0.5).astype(int)
        frac = (idxs - start)
        shifted_arr = np.concatenate((arr[:, 1:], np.zeros((arr.shape[0], 1))), axis=1)
        return arr[:, start] * (1 - frac) + shifted_arr[:, start] * frac

    def round_interpolate_time(self, idxs: np.ndarray, arr: np.ndarray):
        return arr[:, (idxs + 0.5).astype(int)]

    def shift_pitch(self, input_file: str, output_file: str, semitones: float):
        y, self.sr = librosa.load(input_file, sr=None, mono=True)
        X = librosa.stft(y, n_fft=self.w_len, win_length=self.w_len)
        num_freqs, num_frames = X.shape
        scaling = 2 ** (semitones / 12)
        updated_num_frames = np.floor(num_frames * scaling).astype(int)
        updated_t_frames = np.arange(updated_num_frames)
        original_indices = np.minimum(updated_t_frames / scaling, num_frames - 1)
        magnitude = np.abs(X)
        phases = np.angle(X)
        phase_diffs = phases - np.concatenate((np.zeros((num_freqs, 1)), phases[:, :-1]), axis=1)
        phase_diffs = np.mod(phase_diffs, np.pi * 2)
        shifted_magnitude = self.interpolate_time(original_indices, magnitude)
        shifted_phase_diffs = self.interpolate_time(original_indices, phase_diffs)
        unshifted_phases = self.round_interpolate_time(original_indices, phases)
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
        new_waveform = librosa.istft(synth_stft, n_fft=self.w_len, window="hann")
        sf.write(output_file, new_waveform, int(self.sr * scaling))
