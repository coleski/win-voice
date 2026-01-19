"""
Win-Voice - Push-to-Talk Voice Dictation
Hold hotkey to record, release to transcribe and paste.
"""

import os
import sys
import json

# Add CUDA DLL paths before other imports
cuda_paths = [
    os.path.join(sys.prefix, 'Lib', 'site-packages', 'nvidia', 'cublas', 'bin'),
    os.path.join(sys.prefix, 'Lib', 'site-packages', 'nvidia', 'cudnn', 'bin'),
    os.path.dirname(os.path.abspath(__file__)),
]
for p in cuda_paths:
    if os.path.exists(p):
        os.add_dll_directory(p)
        os.environ['PATH'] = p + os.pathsep + os.environ.get('PATH', '')

import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import threading
import time
import pyperclip
import pyautogui
import tkinter as tk
from tkinter import ttk
from pynput import keyboard as pynput_kb
from faster_whisper import WhisperModel

# Config file
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
DEFAULT_CONFIG = {
    "hotkey": "alt",
    "model": "tiny",
    "microphone": None,  # None = default
    "language": "en"
}

pyautogui.FAILSAFE = False
SAMPLE_RATE = 16000

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                cfg = json.load(f)
                return {**DEFAULT_CONFIG, **cfg}
        except:
            pass
    return DEFAULT_CONFIG.copy()

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

def get_microphones():
    """Get list of available microphones"""
    devices = sd.query_devices()
    mics = []
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            mics.append((i, d['name']))
    return mics

def get_hotkey_name(key):
    """Convert pynput key to display name"""
    key_map = {
        'alt': 'Alt',
        'alt_l': 'Left Alt',
        'alt_r': 'Right Alt',
        'ctrl': 'Ctrl',
        'ctrl_l': 'Left Ctrl',
        'ctrl_r': 'Right Ctrl',
        'shift': 'Shift',
        'caps_lock': 'Caps Lock',
        'f1': 'F1', 'f2': 'F2', 'f3': 'F3', 'f4': 'F4',
        'f5': 'F5', 'f6': 'F6', 'f7': 'F7', 'f8': 'F8',
        'f9': 'F9', 'f10': 'F10', 'f11': 'F11', 'f12': 'F12',
    }
    return key_map.get(key, key.title())

class SettingsWindow:
    def __init__(self, config, on_save):
        self.config = config
        self.on_save = on_save
        self.root = tk.Tk()
        self.root.title("Win-Voice Settings")
        self.root.geometry("350x280")
        self.root.resizable(False, False)

        # Hotkey
        ttk.Label(self.root, text="Push-to-Talk Key:").pack(pady=(15,5))
        self.hotkey_var = tk.StringVar(value=get_hotkey_name(config['hotkey']))
        hotkeys = ['Alt', 'Left Alt', 'Right Alt', 'Ctrl', 'Left Ctrl', 'Right Ctrl',
                   'Shift', 'Caps Lock', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8']
        self.hotkey_combo = ttk.Combobox(self.root, textvariable=self.hotkey_var, values=hotkeys, state='readonly', width=25)
        self.hotkey_combo.pack()

        # Model
        ttk.Label(self.root, text="Model (tiny=fast, small=accurate):").pack(pady=(15,5))
        self.model_var = tk.StringVar(value=config['model'])
        models = ['tiny', 'base', 'small', 'medium', 'large-v3']
        self.model_combo = ttk.Combobox(self.root, textvariable=self.model_var, values=models, state='readonly', width=25)
        self.model_combo.pack()

        # Microphone
        ttk.Label(self.root, text="Microphone:").pack(pady=(15,5))
        self.mics = get_microphones()
        mic_names = ['Default'] + [m[1] for m in self.mics]
        self.mic_var = tk.StringVar(value='Default')
        if config['microphone'] is not None:
            for i, name in self.mics:
                if i == config['microphone']:
                    self.mic_var.set(name)
                    break
        self.mic_combo = ttk.Combobox(self.root, textvariable=self.mic_var, values=mic_names, state='readonly', width=25)
        self.mic_combo.pack()

        # Language
        ttk.Label(self.root, text="Language:").pack(pady=(15,5))
        self.lang_var = tk.StringVar(value=config['language'])
        langs = ['en', 'es', 'fr', 'de', 'it', 'pt', 'zh', 'ja', 'ko', 'auto']
        self.lang_combo = ttk.Combobox(self.root, textvariable=self.lang_var, values=langs, state='readonly', width=25)
        self.lang_combo.pack()

        # Save button
        ttk.Button(self.root, text="Save & Start", command=self.save).pack(pady=20)

    def save(self):
        # Convert hotkey display name back to key name
        hotkey_map = {
            'Alt': 'alt', 'Left Alt': 'alt_l', 'Right Alt': 'alt_r',
            'Ctrl': 'ctrl', 'Left Ctrl': 'ctrl_l', 'Right Ctrl': 'ctrl_r',
            'Shift': 'shift', 'Caps Lock': 'caps_lock',
            'F1': 'f1', 'F2': 'f2', 'F3': 'f3', 'F4': 'f4',
            'F5': 'f5', 'F6': 'f6', 'F7': 'f7', 'F8': 'f8',
        }
        self.config['hotkey'] = hotkey_map.get(self.hotkey_var.get(), 'alt')
        self.config['model'] = self.model_var.get()
        self.config['language'] = self.lang_var.get()

        # Get microphone index
        mic_name = self.mic_var.get()
        if mic_name == 'Default':
            self.config['microphone'] = None
        else:
            for i, name in self.mics:
                if name == mic_name:
                    self.config['microphone'] = i
                    break

        save_config(self.config)
        self.root.destroy()
        self.on_save(self.config)

    def run(self):
        self.root.mainloop()

class Overlay:
    def __init__(self):
        self.root = None
        self.label = None

    def setup(self):
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', 0.9)
        self.root.withdraw()
        self.label = tk.Label(self.root, text="", font=("Segoe UI", 12, "bold"), fg="white", padx=15, pady=8)
        self.label.pack()

    def show(self, text, color):
        def _show():
            colors = {"red": "#cc0000", "yellow": "#cc9900", "green": "#00aa00", "gray": "#666666"}
            bg = colors.get(color, "#333333")
            self.label.config(text=text, bg=bg)
            self.root.config(bg=bg)
            self.root.update_idletasks()
            w = self.label.winfo_reqwidth() + 30
            h = self.label.winfo_reqheight() + 16
            x = (self.root.winfo_screenwidth() - w) // 2
            self.root.geometry(f"{w}x{h}+{x}+40")
            self.root.deiconify()
        if self.root:
            self.root.after(0, _show)

    def hide(self):
        if self.root:
            self.root.after(0, lambda: self.root.withdraw())

    def run(self):
        self.setup()
        self.root.mainloop()

class Dictation:
    def __init__(self, config):
        self.config = config
        self.recording = False
        self.processing = False
        self.audio_data = []
        self.model = None
        self.stream = None
        self.overlay = Overlay()
        self.key_pressed = False

    def get_pynput_key(self):
        """Convert config hotkey to pynput key"""
        key_map = {
            'alt': (pynput_kb.Key.alt_l, pynput_kb.Key.alt_r, pynput_kb.Key.alt),
            'alt_l': (pynput_kb.Key.alt_l,),
            'alt_r': (pynput_kb.Key.alt_r,),
            'ctrl': (pynput_kb.Key.ctrl_l, pynput_kb.Key.ctrl_r, pynput_kb.Key.ctrl),
            'ctrl_l': (pynput_kb.Key.ctrl_l,),
            'ctrl_r': (pynput_kb.Key.ctrl_r,),
            'shift': (pynput_kb.Key.shift_l, pynput_kb.Key.shift_r, pynput_kb.Key.shift),
            'caps_lock': (pynput_kb.Key.caps_lock,),
            'f1': (pynput_kb.Key.f1,), 'f2': (pynput_kb.Key.f2,),
            'f3': (pynput_kb.Key.f3,), 'f4': (pynput_kb.Key.f4,),
            'f5': (pynput_kb.Key.f5,), 'f6': (pynput_kb.Key.f6,),
            'f7': (pynput_kb.Key.f7,), 'f8': (pynput_kb.Key.f8,),
        }
        return key_map.get(self.config['hotkey'], (pynput_kb.Key.alt_l, pynput_kb.Key.alt_r))

    def load_model(self):
        model_size = self.config['model']
        print(f"Loading {model_size} model...")
        self.overlay.show("Loading model...", "yellow")
        try:
            self.model = WhisperModel(model_size, device="cuda", compute_type="float16")
            print("GPU loaded!")
        except Exception as e:
            print(f"GPU failed: {e}")
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
            print("CPU loaded")
        self.overlay.show("Ready", "green")
        threading.Timer(1.5, self.overlay.hide).start()

    def audio_callback(self, indata, frames, time_info, status):
        if self.recording:
            self.audio_data.append(indata.copy())

    def start_recording(self):
        if self.recording or self.processing or not self.model:
            return
        self.recording = True
        self.audio_data = []
        self.overlay.show("REC", "red")
        print("Recording...")

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        audio_data = self.audio_data.copy()
        self.audio_data = []

        total_samples = sum(len(a) for a in audio_data)
        if total_samples < int(SAMPLE_RATE * 0.1):
            self.overlay.hide()
            return

        self.processing = True
        threading.Thread(target=self.transcribe, args=(audio_data,)).start()

    def transcribe(self, audio_data):
        self.overlay.show("...", "yellow")

        try:
            audio = np.concatenate(audio_data, axis=0)
        except ValueError:
            self.overlay.hide()
            self.processing = False
            return

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            wav.write(temp_path, SAMPLE_RATE, audio)

        try:
            lang = self.config['language']
            if lang == 'auto':
                lang = None
            segments, _ = self.model.transcribe(temp_path, beam_size=3, language=lang, vad_filter=True)
            text = " ".join([seg.text for seg in segments]).strip()

            if text:
                print(f">>> {text}")
                pyperclip.copy(text)
                time.sleep(0.05)
                pyautogui.hotkey('ctrl', 'v')
                self.overlay.show(text[:40], "green")
                threading.Timer(1.0, self.overlay.hide).start()
            else:
                self.overlay.hide()
        except Exception as e:
            print(f"Error: {e}")
            self.overlay.hide()
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
        self.processing = False

    def on_press(self, key):
        valid_keys = self.get_pynput_key()
        if key in valid_keys:
            if not self.key_pressed:
                self.key_pressed = True
                self.start_recording()

    def on_release(self, key):
        valid_keys = self.get_pynput_key()
        if key in valid_keys:
            if self.key_pressed:
                self.key_pressed = False
                self.stop_recording()

    def run(self):
        # Setup audio stream
        device = self.config['microphone']
        self.stream = sd.InputStream(
            device=device,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
            callback=self.audio_callback
        )
        self.stream.start()

        def init():
            self.load_model()
            listener = pynput_kb.Listener(on_press=self.on_press, on_release=self.on_release)
            listener.start()

        threading.Thread(target=init, daemon=True).start()

        hotkey_name = get_hotkey_name(self.config['hotkey'])
        print("="*40)
        print(f"WIN-VOICE - Hold [{hotkey_name}] to speak")
        print("="*40)

        self.overlay.run()

def main():
    config = load_config()

    # Show settings on first run or if config file is new
    if not os.path.exists(CONFIG_FILE):
        def start_dictation(cfg):
            dictation = Dictation(cfg)
            dictation.run()
        settings = SettingsWindow(config, start_dictation)
        settings.run()
    else:
        dictation = Dictation(config)
        dictation.run()

if __name__ == "__main__":
    main()
