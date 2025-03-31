import pygame
import sys
import threading
import numpy as np
from models.real_time_pitch_detector import detect_pitch, create_audio_stream, fs, CHUNK
from models.pitch_shifter import PitchShifter
import pyaudio
import wave
from datetime import datetime
import os
import subprocess
from pydub import AudioSegment
import simpleaudio as sa

pygame.init()

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 650
FPS = 60
CHART_HEIGHT = 200
CHART_WIDTH = 700
CHART_X = (WINDOW_WIDTH - CHART_WIDTH) // 2
CHART_Y = 220
NUM_BARS = 500

RECORD_FORMAT = pyaudio.paInt16
RECORD_CHANNELS = 1
RECORD_RATE = 44100
RECORD_CHUNK = 1024
RECORD_DURATION = 60

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_GRAY = (245, 247, 250)
DARK_GRAY = (75, 85, 99)
BLUE = (59, 130, 246)
LIGHT_BLUE = (96, 165, 250)
RED = (239, 68, 68)
PURPLE = (147, 51, 234)
GREEN = (34, 197, 94)
BACKGROUND_TOP = (249, 250, 251)
BACKGROUND_BOTTOM = (243, 244, 246)
CHART_GRID = (229, 231, 235)

def create_gradient(color1, color2, height):
    gradient = []
    for i in range(height):
        ratio = i / height
        color = tuple(int(color1[j] + (color2[j] - color1[j]) * ratio) for j in range(3))
        gradient.append(color)
    return gradient

class PitchVisualizer:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Real-time Pitch Detector")
        self.clock = pygame.time.Clock()
        
        self.background_gradient = create_gradient(BACKGROUND_TOP, BACKGROUND_BOTTOM, WINDOW_HEIGHT)
        self.bar_gradient = create_gradient(BLUE, PURPLE, CHART_HEIGHT)
        
        self.pitch_shifter = PitchShifter()
        
        self.button_width = 180
        self.button_height = 45
        self.button_x = WINDOW_WIDTH // 2 - 290
        self.button_y = WINDOW_HEIGHT - 180
        self.button_color = BLUE
        self.button_hover_color = LIGHT_BLUE
        
        self.wav_button_width = 180
        self.wav_button_height = 45
        self.wav_button_x = WINDOW_WIDTH // 2 - 90
        self.wav_button_y = WINDOW_HEIGHT - 180
        self.wav_button_color = GREEN
        self.wav_button_hover_color = (45, 212, 108)
        
        self.shift_button_width = 180
        self.shift_button_height = 45
        self.shift_button_x = WINDOW_WIDTH // 2 + 110
        self.shift_button_y = WINDOW_HEIGHT - 180
        self.shift_button_color = PURPLE
        self.shift_button_hover_color = (167, 71, 254)
        
        self.slider_width = 400
        self.slider_height = 6
        self.slider_x = WINDOW_WIDTH // 2 - self.slider_width // 2
        self.slider_y = WINDOW_HEIGHT - 70
        self.slider_knob_radius = 10
        self.slider_value = 0
        self.is_dragging = False
        self.slider_bg_color = (*DARK_GRAY, 30)
        self.slider_active_color = BLUE
        
        self.chorus_button_width = 150
        self.chorus_button_height = 35
        self.chorus_button_x = self.slider_x - self.chorus_button_width - 20
        self.chorus_button_y = WINDOW_HEIGHT - 80
        self.chorus_button_color = (65, 105, 225)
        self.chorus_button_hover_color = (100, 149, 237)
        
        self.harmonizer_button_width = 150
        self.harmonizer_button_height = 35
        self.harmonizer_button_x = self.slider_x + self.slider_width + 20
        self.harmonizer_button_y = WINDOW_HEIGHT - 80
        self.harmonizer_button_color = (147, 51, 234)
        self.harmonizer_button_hover_color = (167, 71, 254)
        
        self.title_font = pygame.font.Font(None, 48)
        self.note_font = pygame.font.Font(None, 86)
        self.freq_font = pygame.font.Font(None, 36)
        self.label_font = pygame.font.Font(None, 24)
        self.button_font = pygame.font.Font(None, 28)
        self.harmonizer_font = pygame.font.Font(None, 24)
        
        self.is_recording = False
        self.is_recording_wav = False
        self.is_playing_shifted = False
        self.is_playing_chorus = False
        self.is_playing_harmonizer = False
        self.current_play_obj = None
        self.chorus_play_obj = None
        self.harmonizer_play_obj = None
        self.stream = None
        self.update_thread = None
        self.current_note = "--"
        self.current_freq = "--"
        
        self.stream_lock = threading.Lock()
        
        self.spectrum_data = None
        self.max_freq = 4200
        self.detected_freq = None
        
        self.freq_label_step = 600
        
    def draw_text(self, text, font, color, x, y, background=None):
        text_surface = font.render(text, True, color, background)
        text_rect = text_surface.get_rect(center=(x, y))
        self.screen.blit(text_surface, text_rect)
        
    def draw_button(self, text, x, y, width, height, color, hover_color, custom_font=None):
        mouse_pos = pygame.mouse.get_pos()
        button_rect = pygame.Rect(x, y, width, height)
        
        shadow_rect = pygame.Rect(x + 2, y + 2, width, height)
        pygame.draw.rect(self.screen, (*DARK_GRAY, 100), shadow_rect, border_radius=12)
        
        current_color = hover_color if button_rect.collidepoint(mouse_pos) else color
        pygame.draw.rect(self.screen, current_color, button_rect, border_radius=12)
        
        gradient_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        for i in range(height):
            alpha = int(100 * (1 - i/height))
            pygame.draw.line(gradient_surface, (255, 255, 255, alpha), (0, i), (width, i))
        gradient_surface = pygame.transform.scale(gradient_surface, (width, height))
        
        mask = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.rect(mask, (255, 255, 255), mask.get_rect(), border_radius=12)
        gradient_surface.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        self.screen.blit(gradient_surface, button_rect)
        
        font_to_use = custom_font if custom_font else self.button_font
        
        self.draw_text(text, font_to_use, (*BLACK, 50), x + width // 2 + 1, y + height // 2 + 1)
        self.draw_text(text, font_to_use, WHITE, x + width // 2, y + height // 2)
        
        return button_rect
    
    def draw_chart(self):
        chart_surface = pygame.Surface((CHART_WIDTH, CHART_HEIGHT))
        chart_surface.fill(WHITE)
        
        shadow_surface = pygame.Surface((CHART_WIDTH + 4, CHART_HEIGHT + 4), pygame.SRCALPHA)
        pygame.draw.rect(shadow_surface, (*BLACK, 30), shadow_surface.get_rect(), border_radius=15)
        self.screen.blit(shadow_surface, (CHART_X - 2, CHART_Y - 2))
        
        num_vertical_lines = 8
        for i in range(num_vertical_lines):
            x = (CHART_WIDTH * i) // (num_vertical_lines - 1)
            pygame.draw.line(chart_surface, CHART_GRID, (x, 0), (x, CHART_HEIGHT), 1)
        for i in range(5):
            y = (CHART_HEIGHT * i) // 4
            pygame.draw.line(chart_surface, CHART_GRID, (0, y), (CHART_WIDTH, y), 1)
        
        if self.spectrum_data is not None:
            freqs, magnitudes = self.spectrum_data
            
            freq_mask = freqs <= self.max_freq
            freqs = freqs[freq_mask]
            magnitudes = magnitudes[freq_mask]
            
            bar_width = CHART_WIDTH / NUM_BARS
            bar_spacing = 0.3
            
            freq_bins = np.linspace(0, self.max_freq, NUM_BARS + 1)
            
            for i in range(NUM_BARS):
                mask = (freqs >= freq_bins[i]) & (freqs < freq_bins[i + 1])
                if np.any(mask):
                    magnitude = np.max(magnitudes[mask])
                    bar_height = int(magnitude * CHART_HEIGHT)
                    
                    if bar_height > 0:
                        glow_surface = pygame.Surface((int(bar_width), bar_height), pygame.SRCALPHA)
                        for y in range(bar_height):
                            alpha = int(150 * (1 - y/bar_height))
                            color = self.bar_gradient[min(bar_height - y - 1, CHART_HEIGHT-1)]
                            glow_color = (*color, alpha)
                            pygame.draw.line(
                                glow_surface,
                                glow_color,
                                (0, y),
                                (bar_width - bar_spacing, y)
                            )
                        chart_surface.blit(glow_surface, (i * bar_width, CHART_HEIGHT - bar_height))
            
            if self.detected_freq and self.detected_freq <= self.max_freq:
                x = int((self.detected_freq / self.max_freq) * CHART_WIDTH)
                glow_width = 5
                for offset in range(-glow_width, glow_width + 1):
                    alpha = 255 - int(200 * abs(offset) / glow_width)
                    pygame.draw.line(chart_surface, (*RED, alpha),
                                   (x + offset, 0),
                                   (x + offset, CHART_HEIGHT),
                                   2 if offset == 0 else 1)
        
        mask = pygame.Surface((CHART_WIDTH, CHART_HEIGHT), pygame.SRCALPHA)
        pygame.draw.rect(mask, WHITE, mask.get_rect(), border_radius=15)
        
        final_surface = pygame.Surface((CHART_WIDTH, CHART_HEIGHT), pygame.SRCALPHA)
        final_surface.blit(chart_surface, (0, 0))
        final_surface.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        
        self.screen.blit(final_surface, (CHART_X, CHART_Y))
        pygame.draw.rect(self.screen, DARK_GRAY, (CHART_X, CHART_Y, CHART_WIDTH, CHART_HEIGHT), 2, border_radius=15)
        
        for i, freq in enumerate(range(0, self.max_freq + 1, self.freq_label_step)):
            x = CHART_X + (CHART_WIDTH * i * self.freq_label_step) // self.max_freq
            self.draw_text(str(freq), self.label_font, DARK_GRAY, x, CHART_Y + CHART_HEIGHT + 20)
    
    def draw_background(self):
        for y in range(WINDOW_HEIGHT):
            pygame.draw.line(self.screen, self.background_gradient[y], (0, y), (WINDOW_WIDTH, y))
    
    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        self.is_recording = True
        self.stream = create_audio_stream()
        self.update_thread = threading.Thread(target=self.update_display)
        self.update_thread.daemon = True
        self.update_thread.start()
    
    def stop_recording(self):
        with self.stream_lock:
            self.is_recording = False
            if self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except Exception as e:
                    print(f"Error stopping stream: {e}")
                finally:
                    self.stream = None
            
            if self.update_thread and self.update_thread.is_alive():
                self.update_thread.join(timeout=1.0)
            
            self.current_note = "--"
            self.current_freq = "--"
            self.spectrum_data = None
            self.detected_freq = None
    
    def update_display(self):
        while self.is_recording:
            with self.stream_lock:
                if not self.stream:
                    break
                try:
                    note, freq, spectrum = detect_pitch(self.stream)
                    if note and freq and spectrum:
                        self.current_note = note
                        self.current_freq = f"{freq:.1f}"
                        self.detected_freq = freq
                        self.spectrum_data = spectrum
                except Exception as e:
                    print(f"Error in update_display: {e}")
                    break
    
    def record_to_wav(self):
        if self.is_recording_wav:
            return
            
        self.is_recording_wav = True
        output_file = "recording.wav"
        
        audio = pyaudio.PyAudio()
        stream = audio.open(format=RECORD_FORMAT, channels=RECORD_CHANNELS, rate=RECORD_RATE,
                          input=True, frames_per_buffer=RECORD_CHUNK)
        
        print(f"Recording to {output_file}...")
        frames = []
        
        total_chunks = int(RECORD_RATE / RECORD_CHUNK * RECORD_DURATION)
        
        for _ in range(total_chunks):
            if not self.is_recording_wav:
                break
            data = stream.read(RECORD_CHUNK)
            frames.append(data)
        
        print("Recording complete.")
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(RECORD_CHANNELS)
            wf.setsampwidth(audio.get_sample_size(RECORD_FORMAT))
            wf.setframerate(RECORD_RATE)
            wf.writeframes(b''.join(frames))
        
        print(f"Audio saved to {output_file}")
        self.is_recording_wav = False

    def draw_slider(self):
        track_height = self.slider_height
        track_rect = pygame.Rect(self.slider_x, self.slider_y - track_height//2, 
                               self.slider_width, track_height)
        
        pygame.draw.rect(self.screen, self.slider_bg_color, track_rect, border_radius=4)
        
        value_range = 24
        center_x = self.slider_x + self.slider_width // 2
        knob_x = self.slider_x + int((self.slider_value + 12) * self.slider_width / value_range)
        
        if knob_x != center_x:
            active_rect = pygame.Rect(min(center_x, knob_x), self.slider_y - track_height//2,
                                    abs(knob_x - center_x), track_height)
            active_color = (*self.slider_active_color, 150)
            pygame.draw.rect(self.screen, active_color, active_rect, border_radius=4)
        
        center_marker_height = 12
        pygame.draw.rect(self.screen, (*DARK_GRAY, 100),
                        (center_x - 1, self.slider_y - center_marker_height//2,
                         2, center_marker_height))
        
        knob_y = self.slider_y
        
        shadow_radius = self.slider_knob_radius + 2
        pygame.draw.circle(self.screen, (*BLACK, 30), (knob_x + 1, knob_y + 1), shadow_radius)
        
        knob_color = LIGHT_BLUE if self.is_dragging else BLUE
        pygame.draw.circle(self.screen, knob_color, (knob_x, knob_y), self.slider_knob_radius)
        
        highlight_radius = self.slider_knob_radius - 2
        pygame.draw.circle(self.screen, (*WHITE, 50), 
                         (knob_x - 1, knob_y - 1), 
                         highlight_radius)
        
        label_y = self.slider_y - 25
        value_text = f"{self.slider_value:+d} semitones"
        
        text_surface = self.button_font.render(value_text, True, DARK_GRAY)
        text_rect = text_surface.get_rect(center=(WINDOW_WIDTH // 2, label_y))
        padding = 10
        bg_rect = pygame.Rect(text_rect.x - padding, text_rect.y - padding//2,
                            text_rect.width + padding * 2, text_rect.height + padding)
        pygame.draw.rect(self.screen, (*WHITE, 150), bg_rect, border_radius=6)
        
        self.draw_text(value_text, self.button_font, DARK_GRAY,
                      WINDOW_WIDTH // 2, label_y)

    def handle_slider_interaction(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            value_range = 24
            knob_x = self.slider_x + int((self.slider_value + 12) * self.slider_width / value_range)
            knob_y = self.slider_y + self.slider_height // 2
            knob_rect = pygame.Rect(knob_x - self.slider_knob_radius, 
                                  knob_y - self.slider_knob_radius,
                                  self.slider_knob_radius * 2,
                                  self.slider_knob_radius * 2)
            if knob_rect.collidepoint(event.pos):
                self.is_dragging = True
        
        elif event.type == pygame.MOUSEBUTTONUP:
            self.is_dragging = False
        
        elif event.type == pygame.MOUSEMOTION and self.is_dragging:
            x = max(self.slider_x, min(event.pos[0], self.slider_x + self.slider_width))
            value_range = 24
            self.slider_value = round((x - self.slider_x) * value_range / self.slider_width - 12)
            self.slider_value = max(-12, min(12, self.slider_value))

    def shift_and_play_audio(self):
        if not os.path.exists("recording.wav"):
            print("No recording found. Please record audio first.")
            return
            
        try:
            if self.is_playing_shifted:
                if self.current_play_obj:
                    self.current_play_obj.stop()
                self.current_play_obj = None
                self.is_playing_shifted = False
                return
            
            self.pitch_shifter.shift_pitch("recording.wav", "shifted_temp.wav", self.slider_value)
            
            try:
                audio = AudioSegment.from_wav("shifted_temp.wav")
                audio = audio.set_frame_rate(RECORD_RATE)
                audio.export("shifted.wav", format="wav")
                
                if os.path.exists("shifted_temp.wav"):
                    try:
                        os.remove("shifted_temp.wav")
                    except:
                        pass
                
                wave_obj = sa.WaveObject.from_wave_file("shifted.wav")
                self.current_play_obj = wave_obj.play()
                self.is_playing_shifted = True
                
                def monitor_playback():
                    try:
                        if self.current_play_obj:
                            self.current_play_obj.wait_done()
                    except Exception as e:
                        print(f"Error in playback monitoring: {e}")
                    finally:
                        self.is_playing_shifted = False
                        self.current_play_obj = None
                        if os.path.exists("shifted_temp.wav"):
                            try:
                                os.remove("shifted_temp.wav")
                            except:
                                pass
                
                threading.Thread(target=monitor_playback, daemon=True).start()
                
            except Exception as e:
                print(f"Error playing shifted audio: {e}")
                import traceback
                traceback.print_exc()
                self.is_playing_shifted = False
                self.current_play_obj = None
                if os.path.exists("shifted_temp.wav"):
                    try:
                        os.remove("shifted_temp.wav")
                    except:
                        pass
                
        except Exception as e:
            print(f"Error in shift_and_play_audio: {e}")
            import traceback
            traceback.print_exc()
            self.is_playing_shifted = False
            self.current_play_obj = None

    def create_harmonizer_effect(self):
        if not os.path.exists("recording.wav"):
            print("No recording found. Please record audio first.")
            return
            
        try:
            if self.is_playing_harmonizer:
                if self.harmonizer_play_obj:
                    self.harmonizer_play_obj.stop()
                self.harmonizer_play_obj = None
                self.is_playing_harmonizer = False
                return

            self.pitch_shifter.shift_pitch("recording.wav", "shifted_temp.wav", self.slider_value)
            try:
                audio = AudioSegment.from_wav("shifted_temp.wav")
                audio = audio.set_frame_rate(RECORD_RATE)
                audio.export("shifted.wav", format="wav")
                
                if os.path.exists("shifted_temp.wav"):
                    try:
                        os.remove("shifted_temp.wav")
                    except:
                        pass
            except Exception as e:
                print(f"Error generating shifted audio: {e}")
                return

            self.pitch_shifter.shift_pitch("shifted.wav", "lower_harmony.wav", -3.02)
            self.pitch_shifter.shift_pitch("shifted.wav", "upper_harmony.wav", 3.98)
            self.pitch_shifter.shift_pitch("shifted.wav", "fifth_harmony.wav", 7.02)
            self.pitch_shifter.shift_pitch("shifted.wav", "octave.wav", 11.98)
            self.pitch_shifter.shift_pitch("shifted.wav", "lower_detune.wav", -3.08)
            self.pitch_shifter.shift_pitch("shifted.wav", "upper_detune.wav", 4.04)

            try:
                original = AudioSegment.from_wav("shifted.wav").set_frame_rate(RECORD_RATE)
                lower = AudioSegment.from_wav("lower_harmony.wav").set_frame_rate(RECORD_RATE)
                upper = AudioSegment.from_wav("upper_harmony.wav").set_frame_rate(RECORD_RATE)
                fifth = AudioSegment.from_wav("fifth_harmony.wav").set_frame_rate(RECORD_RATE)
                octave = AudioSegment.from_wav("octave.wav").set_frame_rate(RECORD_RATE)
                lower_detune = AudioSegment.from_wav("lower_detune.wav").set_frame_rate(RECORD_RATE)
                upper_detune = AudioSegment.from_wav("upper_detune.wav").set_frame_rate(RECORD_RATE)

                silence_10ms = AudioSegment.silent(duration=10)
                silence_15ms = AudioSegment.silent(duration=15)
                silence_20ms = AudioSegment.silent(duration=20)
                silence_25ms = AudioSegment.silent(duration=25)
                silence_30ms = AudioSegment.silent(duration=30)

                lower = silence_10ms + lower
                upper = silence_15ms + upper
                fifth = silence_20ms + fifth
                octave = silence_25ms + octave
                lower_detune = silence_20ms + lower_detune
                upper_detune = silence_30ms + upper_detune

                lower = lower.pan(-0.3)
                upper = upper.pan(0.3)
                fifth = fifth.pan(-0.15)
                octave = octave.pan(0.15)
                lower_detune = lower_detune.pan(-0.4)
                upper_detune = upper_detune.pan(0.4)

                original = original - 4
                lower = lower - 10
                upper = upper - 9
                fifth = fifth - 11
                octave = octave - 13
                lower_detune = lower_detune - 15
                upper_detune = upper_detune - 15

                lower = lower.overlay(lower_detune)
                upper = upper.overlay(upper_detune)

                combined_audio = original
                combined_audio = combined_audio.overlay(lower)
                combined_audio = combined_audio.overlay(upper)
                combined_audio = combined_audio.overlay(fifth)
                combined_audio = combined_audio.overlay(octave)

                reverb_copies = []
                for i in range(3):
                    delay_ms = 40 + (i * 20)
                    volume_reduction = 20 + (i * 5)
                    reverb_copy = combined_audio - volume_reduction
                    reverb_copy = AudioSegment.silent(duration=delay_ms) + reverb_copy
                    reverb_copies.append(reverb_copy)

                for reverb in reverb_copies:
                    combined_audio = combined_audio.overlay(reverb)

                combined_audio.export("harmonized.wav", format="wav", 
                                    parameters=["-ar", str(RECORD_RATE), "-q:a", "0"])

                wave_obj = sa.WaveObject.from_wave_file("harmonized.wav")
                self.harmonizer_play_obj = wave_obj.play()
                self.is_playing_harmonizer = True
                
                def monitor_playback():
                    if self.harmonizer_play_obj:
                        self.harmonizer_play_obj.wait_done()
                        self.is_playing_harmonizer = False
                        self.harmonizer_play_obj = None
                
                threading.Thread(target=monitor_playback, daemon=True).start()
                
            except Exception as e:
                print(f"Error mixing audio: {e}")
                import traceback
                traceback.print_exc()
                self.is_playing_harmonizer = False
                self.harmonizer_play_obj = None
            finally:
                temp_files = ["lower_harmony.wav", "upper_harmony.wav", "fifth_harmony.wav", 
                            "octave.wav", "lower_detune.wav", "upper_detune.wav"]
                for file in temp_files:
                    if os.path.exists(file):
                        try:
                            os.remove(file)
                        except:
                            pass
                
        except Exception as e:
            print(f"Error creating harmonizer effect: {e}")
            self.is_playing_harmonizer = False
            self.harmonizer_play_obj = None

    def create_chorus_effect(self):
        if not os.path.exists("recording.wav"):
            print("No recording found. Please record audio first.")
            return
            
        try:
            if self.is_playing_chorus:
                if self.chorus_play_obj:
                    self.chorus_play_obj.stop()
                self.chorus_play_obj = None
                self.is_playing_chorus = False
                return

            self.pitch_shifter.shift_pitch("recording.wav", "shifted_temp.wav", self.slider_value)
            try:
                audio = AudioSegment.from_wav("shifted_temp.wav")
                audio = audio.set_frame_rate(RECORD_RATE)
                audio.export("shifted.wav", format="wav")
                
                if os.path.exists("shifted_temp.wav"):
                    try:
                        os.remove("shifted_temp.wav")
                    except:
                        pass
            except Exception as e:
                print(f"Error generating shifted audio: {e}")
                return

            self.pitch_shifter.shift_pitch("shifted.wav", "chorus1.wav", 0.15)
            self.pitch_shifter.shift_pitch("shifted.wav", "chorus2.wav", -0.15)
            self.pitch_shifter.shift_pitch("shifted.wav", "chorus3.wav", 0.08)
            self.pitch_shifter.shift_pitch("shifted.wav", "chorus4.wav", -0.08)
            self.pitch_shifter.shift_pitch("shifted.wav", "chorus5.wav", 0.2)
            
            try:
                original = AudioSegment.from_wav("shifted.wav").set_frame_rate(RECORD_RATE)
                chorus1 = AudioSegment.from_wav("chorus1.wav").set_frame_rate(RECORD_RATE)
                chorus2 = AudioSegment.from_wav("chorus2.wav").set_frame_rate(RECORD_RATE)
                chorus3 = AudioSegment.from_wav("chorus3.wav").set_frame_rate(RECORD_RATE)
                
                delay1 = 20
                delay2 = 30
                delay3 = 15
                
                silence1 = AudioSegment.silent(duration=delay1)
                silence2 = AudioSegment.silent(duration=delay2)
                silence3 = AudioSegment.silent(duration=delay3)
                
                chorus1 = silence1 + chorus1
                chorus2 = silence2 + chorus2
                chorus3 = silence3 + chorus3
                
                original = original - 3 
                chorus1 = chorus1 - 4
                chorus2 = chorus2 - 4
                chorus3 = chorus3 - 6
                
                combined_audio = original.overlay(chorus1)
                combined_audio = combined_audio.overlay(chorus2)
                combined_audio = combined_audio.overlay(chorus3)
                
                combined_audio.export("chorus.wav", format="wav", parameters=["-ar", str(RECORD_RATE)])
                
                wave_obj = sa.WaveObject.from_wave_file("chorus.wav")
                self.chorus_play_obj = wave_obj.play()
                self.is_playing_chorus = True
                
                def monitor_playback():
                    if self.chorus_play_obj:
                        self.chorus_play_obj.wait_done()
                        self.is_playing_chorus = False
                        self.chorus_play_obj = None
                
                threading.Thread(target=monitor_playback, daemon=True).start()
                
            except Exception as e:
                print(f"Error creating chorus effect: {e}")
                import traceback
                traceback.print_exc()
                self.is_playing_chorus = False
                self.chorus_play_obj = None
            finally:
                temp_files = ["chorus1.wav", "chorus2.wav", "chorus3.wav", "chorus4.wav", "chorus5.wav"]
                for file in temp_files:
                    if os.path.exists(file):
                        try:
                            os.remove(file)
                        except:
                            pass
                
        except Exception as e:
            print(f"Error creating chorus effect: {e}")
            self.is_playing_chorus = False
            self.chorus_play_obj = None

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pitch_button_rect = pygame.Rect(self.button_x, self.button_y, self.button_width, self.button_height)
                    if pitch_button_rect.collidepoint(event.pos):
                        self.toggle_recording()
                    
                    wav_button_rect = pygame.Rect(self.wav_button_x, self.wav_button_y, self.wav_button_width, self.wav_button_height)
                    if wav_button_rect.collidepoint(event.pos):
                        if self.is_recording_wav:
                            self.is_recording_wav = False
                        else:
                            threading.Thread(target=self.record_to_wav).start()
                    
                    shift_button_rect = pygame.Rect(self.shift_button_x, self.shift_button_y, 
                                                  self.shift_button_width, self.shift_button_height)
                    if shift_button_rect.collidepoint(event.pos):
                        threading.Thread(target=self.shift_and_play_audio).start()
                    
                    chorus_button_rect = pygame.Rect(self.chorus_button_x, self.chorus_button_y,
                                                   self.chorus_button_width, self.chorus_button_height)
                    if chorus_button_rect.collidepoint(event.pos):
                        threading.Thread(target=self.create_chorus_effect).start()
                    
                    harmonizer_button_rect = pygame.Rect(self.harmonizer_button_x, self.harmonizer_button_y,
                                                       self.harmonizer_button_width, self.harmonizer_button_height)
                    if harmonizer_button_rect.collidepoint(event.pos):
                        threading.Thread(target=self.create_harmonizer_effect).start()
                
                self.handle_slider_interaction(event)
            
            self.draw_background()
            
            shadow_offset = 2
            for offset in range(1, 4):
                alpha = 100 - offset * 25
                self.draw_text("Real-time Pitch Detector", self.title_font, (*DARK_GRAY, alpha), 
                             WINDOW_WIDTH // 2 + offset, 52 + offset)
            self.draw_text("Real-time Pitch Detector", self.title_font, BLUE, WINDOW_WIDTH // 2, 50)
            
            note_bg = pygame.Surface((320, 100), pygame.SRCALPHA)
            pygame.draw.rect(note_bg, (*WHITE, 230), note_bg.get_rect(), border_radius=15)
            self.screen.blit(note_bg, (WINDOW_WIDTH // 2 - 160, 100))
            self.draw_text(f"Note: {self.current_note}", self.note_font, BLUE, WINDOW_WIDTH // 2, 150)
            
            self.draw_text(f"Frequency: {self.current_freq} Hz", self.freq_font, (*DARK_GRAY, 100), 
                         WINDOW_WIDTH // 2 + 1, 191)
            self.draw_text(f"Frequency: {self.current_freq} Hz", self.freq_font, DARK_GRAY, 
                         WINDOW_WIDTH // 2, 190)
            
            self.draw_chart()
            
            self.draw_slider()
            
            pitch_button_text = "Stop Recording" if self.is_recording else "Start Recording"
            self.draw_button(pitch_button_text, self.button_x, self.button_y, self.button_width, self.button_height,
                           self.button_color, self.button_hover_color)
            
            wav_button_text = "Recording..." if self.is_recording_wav else "Record to WAV"
            self.draw_button(wav_button_text, self.wav_button_x, self.wav_button_y, self.wav_button_width, self.wav_button_height,
                           self.wav_button_color, self.wav_button_hover_color)
            
            shift_button_text = "Stop Playing" if self.is_playing_shifted else "Play Audio"
            self.draw_button(shift_button_text, self.shift_button_x, self.shift_button_y,
                           self.shift_button_width, self.shift_button_height,
                           self.shift_button_color, self.shift_button_hover_color)
            
            chorus_button_text = "Stop Chorus" if self.is_playing_chorus else "Chorus Effect"
            self.draw_button(chorus_button_text, self.chorus_button_x, self.chorus_button_y,
                           self.chorus_button_width, self.chorus_button_height,
                           self.chorus_button_color, self.chorus_button_hover_color,
                           custom_font=self.harmonizer_font)
            
            harmonizer_button_text = "Stop Harmonizer" if self.is_playing_harmonizer else "Harmonizer Effect"
            self.draw_button(harmonizer_button_text, self.harmonizer_button_x, self.harmonizer_button_y,
                           self.harmonizer_button_width, self.harmonizer_button_height,
                           self.harmonizer_button_color, self.harmonizer_button_hover_color,
                           custom_font=self.harmonizer_font)
            
            pygame.display.flip()
            self.clock.tick(FPS)
        
        self.stop_recording()
        pygame.quit()
        sys.exit()

def main():
    app = PitchVisualizer()
    app.run()

if __name__ == "__main__":
    main() 