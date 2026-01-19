"""
Win-Voice - Push-to-Talk Voice Dictation
System tray app - click icon to open settings.
"""

import os
import sys
import json

# Get app directory (works for script and exe)
if getattr(sys, 'frozen', False):
    APP_DIR = os.path.dirname(sys.executable)
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Add CUDA DLL paths
cuda_paths = [
    os.path.join(sys.prefix, 'Lib', 'site-packages', 'nvidia', 'cublas', 'bin'),
    os.path.join(sys.prefix, 'Lib', 'site-packages', 'nvidia', 'cudnn', 'bin'),
    APP_DIR,
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
import ctypes
from ctypes import wintypes
from PIL import Image, ImageDraw
import pystray
from faster_whisper import WhisperModel

# Windows key codes
VK_CODES = {
    'alt': 0x12,  # VK_MENU
    'ctrl': 0x11, # VK_CONTROL
    'shift': 0x10, # VK_SHIFT
    'f1': 0x70, 'f2': 0x71, 'f3': 0x72, 'f4': 0x73,
    'f5': 0x74, 'f6': 0x75, 'f7': 0x76, 'f8': 0x77,
}

CONFIG_FILE = os.path.join(APP_DIR, "config.json")
DEFAULT_CONFIG = {"hotkey": "alt", "model": "tiny", "microphone": None, "language": "en"}
SAMPLE_RATE = 16000
pyautogui.FAILSAFE = False

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        except: pass
    return DEFAULT_CONFIG.copy()

def save_config(cfg):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(cfg, f, indent=2)

def get_microphones():
    return [(i, d['name']) for i, d in enumerate(sd.query_devices()) if d['max_input_channels'] > 0]

def create_icon(color):
    img = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    c = {"green": (0, 200, 0), "red": (200, 0, 0), "yellow": (200, 200, 0)}.get(color, (128, 128, 128))
    draw.ellipse([20, 8, 44, 40], fill=c)
    draw.rectangle([22, 24, 42, 44], fill=c)
    draw.arc([16, 28, 48, 52], 0, 180, fill=c, width=3)
    draw.line([32, 52, 32, 58], fill=c, width=3)
    draw.line([22, 58, 42, 58], fill=c, width=3)
    return img

class WinVoice:
    def __init__(self):
        self.config = load_config()
        self.recording = False
        self.processing = False
        self.audio_data = []
        self.model = None
        self.stream = None
        self.tray = None
        self.key_pressed = False
        self.running = True

    def get_vk_code(self):
        return VK_CODES.get(self.config['hotkey'], 0x12)

    def load_model(self):
        print(f"Loading {self.config['model']} model...")
        try:
            self.model = WhisperModel(self.config['model'], device="cuda", compute_type="float16")
            print("GPU loaded!")
        except Exception as e:
            print(f"GPU failed: {e}, using CPU")
            self.model = WhisperModel(self.config['model'], device="cpu", compute_type="int8")

    def audio_callback(self, indata, frames, time_info, status):
        if self.recording:
            self.audio_data.append(indata.copy())

    def start_recording(self):
        if self.recording or self.processing or not self.model:
            return
        self.recording = True
        self.audio_data = []
        if self.tray:
            self.tray.icon = create_icon("red")
        print("Recording...")

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        audio_data = self.audio_data.copy()
        self.audio_data = []

        if self.tray:
            self.tray.icon = create_icon("yellow")

        total = sum(len(a) for a in audio_data)
        print(f"Audio samples: {total} ({total/SAMPLE_RATE:.2f}s)")
        if total < int(SAMPLE_RATE * 0.1):
            print("Too short, skipping")
            if self.tray:
                self.tray.icon = create_icon("green")
            return

        self.processing = True
        threading.Thread(target=self.transcribe, args=(audio_data,), daemon=True).start()

    def transcribe(self, audio_data):
        print("Transcribing...")
        try:
            audio = np.concatenate(audio_data, axis=0)
        except Exception as e:
            print(f"Concat error: {e}")
            self.processing = False
            if self.tray:
                self.tray.icon = create_icon("green")
            return

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            wav.write(temp_path, SAMPLE_RATE, audio)
        print(f"Saved to {temp_path}")

        try:
            lang = self.config['language'] if self.config['language'] != 'auto' else None
            print(f"Calling model.transcribe(lang={lang})...")
            segments, _ = self.model.transcribe(temp_path, beam_size=3, language=lang, vad_filter=True)
            text = " ".join([s.text for s in segments]).strip()
            print(f"Result: '{text}'")
            if text:
                print(f">>> {text}")
                pyperclip.copy(text)
                time.sleep(0.05)
                pyautogui.hotkey('ctrl', 'v')
        except Exception as e:
            print(f"Transcribe error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try: os.unlink(temp_path)
            except: pass

        self.processing = False
        if self.tray:
            self.tray.icon = create_icon("green")

    def key_poll_loop(self):
        """Poll for hotkey state using GetAsyncKeyState"""
        user32 = ctypes.windll.user32
        vk = self.get_vk_code()
        print(f"Polling for VK code: {hex(vk)}")

        while self.running:
            # GetAsyncKeyState returns negative if key is pressed
            state = user32.GetAsyncKeyState(vk)
            is_pressed = state & 0x8000

            if is_pressed and not self.key_pressed:
                print(f"Key DOWN")
                self.key_pressed = True
                self.start_recording()
            elif not is_pressed and self.key_pressed:
                print(f"Key UP")
                self.key_pressed = False
                self.stop_recording()

            time.sleep(0.01)  # 10ms polling

    def show_settings(self):
        def save_and_close():
            hk_map = {'Alt': 'alt', 'Ctrl': 'ctrl', 'Shift': 'shift',
                      'F1': 'f1', 'F2': 'f2', 'F3': 'f3', 'F4': 'f4', 'F5': 'f5', 'F6': 'f6', 'F7': 'f7', 'F8': 'f8'}
            self.config['hotkey'] = hk_map.get(hk_var.get(), 'alt')
            self.config['model'] = model_var.get()
            self.config['language'] = lang_var.get()
            mic = mic_var.get()
            self.config['microphone'] = None if mic == 'Default' else next((i for i, n in mics if n == mic), None)
            save_config(self.config)
            win.destroy()
            # Reload model if changed
            print("Settings saved. Restart to apply model changes.")

        win = tk.Tk()
        win.title("Win-Voice Settings")
        win.geometry("320x320")
        win.resizable(False, False)
        win.attributes('-topmost', True)

        ttk.Label(win, text="Push-to-Talk Key:").pack(pady=(15, 5))
        hk_var = tk.StringVar(value=self.config['hotkey'].title())
        ttk.Combobox(win, textvariable=hk_var, values=['Alt', 'Ctrl', 'Shift', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8'], state='readonly', width=20).pack()

        ttk.Label(win, text="Model:").pack(pady=(10, 5))
        model_var = tk.StringVar(value=self.config['model'])
        ttk.Combobox(win, textvariable=model_var, values=['tiny', 'base', 'small', 'medium', 'large-v3'], state='readonly', width=20).pack()

        ttk.Label(win, text="Microphone:").pack(pady=(10, 5))
        mics = get_microphones()
        mic_names = ['Default'] + [n for _, n in mics]
        mic_var = tk.StringVar(value='Default')
        if self.config['microphone']:
            for i, n in mics:
                if i == self.config['microphone']:
                    mic_var.set(n)
        ttk.Combobox(win, textvariable=mic_var, values=mic_names, state='readonly', width=20).pack()

        ttk.Label(win, text="Language:").pack(pady=(10, 5))
        lang_var = tk.StringVar(value=self.config['language'])
        ttk.Combobox(win, textvariable=lang_var, values=['en', 'es', 'fr', 'de', 'auto'], state='readonly', width=20).pack()

        ttk.Button(win, text="Save", command=save_and_close).pack(pady=15)
        win.mainloop()

    def quit_app(self):
        self.running = False
        if self.stream:
            self.stream.stop()
        if self.tray:
            self.tray.stop()
        os._exit(0)

    def run(self):
        # Start audio
        self.stream = sd.InputStream(device=self.config['microphone'], samplerate=SAMPLE_RATE,
                                      channels=1, dtype=np.float32, callback=self.audio_callback)
        self.stream.start()

        # Start keyboard polling thread
        threading.Thread(target=self.key_poll_loop, daemon=True).start()

        # Load model in background
        threading.Thread(target=self.load_model, daemon=True).start()

        # Create tray (Settings is default action on left-click)
        menu = pystray.Menu(
            pystray.MenuItem("Settings", lambda: threading.Thread(target=self.show_settings, daemon=True).start(), default=True),
            pystray.MenuItem("Quit", self.quit_app)
        )
        self.tray = pystray.Icon("WinVoice", create_icon("yellow"), "Win-Voice - Loading...", menu)

        def on_tray_ready(icon):
            icon.visible = True
            # Update to green after model loads
            def wait_for_model():
                while self.model is None:
                    time.sleep(0.5)
                hk = self.config['hotkey'].title()
                icon.title = f"Win-Voice - Hold {hk}"
                icon.icon = create_icon("green")
            threading.Thread(target=wait_for_model, daemon=True).start()

        self.tray.run(on_tray_ready)

if __name__ == "__main__":
    save_config(load_config())  # Ensure config exists
    WinVoice().run()
