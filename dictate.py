"""
Win-Voice - Push-to-Talk Voice Dictation
System tray app - click icon to open settings.
"""

import os
import sys
import json
import winsound
import queue

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

# Sound file paths
SOUNDS_DIR = os.path.join(APP_DIR, "sounds")
SOUND_START = os.path.join(SOUNDS_DIR, "start.wav")
SOUND_STOP = os.path.join(SOUNDS_DIR, "stop.wav")
SOUND_PASTE = os.path.join(SOUNDS_DIR, "paste.wav")

def play_sound(sound_path):
    """Play a sound file asynchronously (non-blocking)"""
    try:
        if os.path.exists(sound_path):
            print(f"Playing sound: {sound_path}")
            winsound.PlaySound(sound_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        else:
            print(f"Sound file not found: {sound_path}")
    except Exception as e:
        print(f"Sound error: {e}")

class OverlayOrb:
    """Floating orb overlay with smiley face that shows recording state"""

    # Colors for different states
    COLORS = {
        'loading': '#FFD700',   # Yellow/gold
        'ready': '#00C853',     # Green
        'recording': '#FF1744', # Red
        'processing': '#FFD700' # Yellow
    }

    def __init__(self):
        self.root = None
        self.canvas = None
        self.state = 'loading'
        self.eye_spread = 0  # Animation: how far eyes spread apart
        self.pulse_scale = 1.0  # Animation: pulse scale
        self.animation_running = False
        self.command_queue = queue.Queue()  # Thread-safe command queue
        self._setup_window()

    def _setup_window(self):
        """Create the overlay window"""
        self.root = tk.Toplevel()
        self.root.withdraw()  # Start hidden

        # Window properties: borderless, always on top
        self.root.overrideredirect(True)
        self.root.attributes('-topmost', True)

        # Use a specific color for transparency (magenta is common)
        self.transparent_color = '#010101'  # Near-black
        self.root.attributes('-transparentcolor', self.transparent_color)

        # Prevent showing in taskbar (Windows-specific)
        self.root.attributes('-toolwindow', True)

        # Size and position (center of screen)
        size = 100
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = (screen_w - size) // 2
        y = (screen_h - size) // 2
        self.root.geometry(f"{size}x{size}+{x}+{y}")

        # Canvas for drawing - use transparent color as background
        self.canvas = tk.Canvas(self.root, width=size, height=size,
                                bg=self.transparent_color, highlightthickness=0)
        self.canvas.pack()
        print(f"Overlay window created at {x},{y}")

    def _draw_face(self):
        """Draw the smiley face based on current state"""
        self.canvas.delete('all')

        size = 100
        center = size // 2
        color = self.COLORS.get(self.state, self.COLORS['loading'])

        # Apply pulse scale for recording animation
        radius = int(38 * self.pulse_scale)

        # Main face circle
        self.canvas.create_oval(
            center - radius, center - radius,
            center + radius, center + radius,
            fill=color, outline=''
        )

        # Eye parameters
        eye_y = center - 8
        base_eye_x = 12
        eye_spread = base_eye_x + self.eye_spread
        eye_size = 6

        # Left eye
        self.canvas.create_oval(
            center - eye_spread - eye_size, eye_y - eye_size,
            center - eye_spread + eye_size, eye_y + eye_size,
            fill='black', outline=''
        )

        # Right eye
        self.canvas.create_oval(
            center + eye_spread - eye_size, eye_y - eye_size,
            center + eye_spread + eye_size, eye_y + eye_size,
            fill='black', outline=''
        )

        # Mouth based on state
        mouth_y = center + 10
        if self.state == 'recording':
            # Open mouth (recording) - "O" shape
            self.canvas.create_oval(
                center - 8, mouth_y - 6,
                center + 8, mouth_y + 8,
                fill='black', outline=''
            )
        elif self.state == 'processing':
            # Thinking mouth - small line
            self.canvas.create_line(
                center - 8, mouth_y + 2,
                center + 8, mouth_y + 2,
                fill='black', width=3
            )
        else:
            # Happy smile arc
            self.canvas.create_arc(
                center - 15, mouth_y - 12,
                center + 15, mouth_y + 12,
                start=200, extent=140,
                style='arc', outline='black', width=3
            )

    def set_state(self, state):
        """Queue state change (thread-safe)"""
        self.command_queue.put(('set_state', state))

    def show(self):
        """Queue show command (thread-safe)"""
        self.command_queue.put(('show', None))

    def hide(self):
        """Queue hide command (thread-safe)"""
        self.command_queue.put(('hide', None))

    def process_commands(self):
        """Process queued commands - call from main thread"""
        try:
            while True:
                cmd, arg = self.command_queue.get_nowait()
                if cmd == 'set_state':
                    self._do_set_state(arg)
                elif cmd == 'show':
                    self._do_show()
                elif cmd == 'hide':
                    self._do_hide()
        except queue.Empty:
            pass

    def _do_set_state(self, state):
        """Actually update state (main thread only)"""
        self.state = state
        if state == 'recording':
            self.eye_spread = 6
            self.animation_running = True
            self._animate_pulse(True)
        else:
            self.eye_spread = 0
            self.pulse_scale = 1.0
            self.animation_running = False
            self._draw_face()

    def _do_show(self):
        """Actually show overlay (main thread only)"""
        print("Overlay _do_show called")
        self._draw_face()
        self.root.deiconify()
        self.root.lift()
        self.root.attributes('-topmost', True)  # Ensure on top
        print(f"Overlay visible, geometry: {self.root.geometry()}")

    def _do_hide(self):
        """Actually hide overlay (main thread only)"""
        self.animation_running = False
        self.root.withdraw()

    def _animate_pulse(self, growing=True):
        """Animate pulse effect (main thread only)"""
        if not self.animation_running or self.state != 'recording':
            self.pulse_scale = 1.0
            self._draw_face()
            return

        if growing:
            self.pulse_scale += 0.01
            if self.pulse_scale >= 1.05:
                growing = False
        else:
            self.pulse_scale -= 0.01
            if self.pulse_scale <= 0.95:
                growing = True

        self._draw_face()

        if self.root and self.animation_running:
            self.root.after(50, lambda: self._animate_pulse(growing))

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
        self.overlay = None  # Initialized after tkinter root exists

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
        # Show overlay and play start sound
        print(f"Overlay exists: {self.overlay is not None}")
        if self.overlay:
            print("Setting overlay state to recording and showing...")
            self.overlay.set_state('recording')
            self.overlay.show()
        play_sound(SOUND_START)
        print("Recording...")

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        audio_data = self.audio_data.copy()
        self.audio_data = []

        if self.tray:
            self.tray.icon = create_icon("yellow")

        # Play stop sound and set overlay to processing
        play_sound(SOUND_STOP)
        if self.overlay:
            self.overlay.set_state('processing')

        total = sum(len(a) for a in audio_data)
        print(f"Audio samples: {total} ({total/SAMPLE_RATE:.2f}s)")
        if total < int(SAMPLE_RATE * 0.1):
            print("Too short, skipping")
            if self.tray:
                self.tray.icon = create_icon("green")
            if self.overlay:
                self.overlay.hide()
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
            if self.overlay:
                self.overlay.hide()
            return

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            wav.write(temp_path, SAMPLE_RATE, audio)
        print(f"Saved to {temp_path}")

        pasted = False
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
                pasted = True
                play_sound(SOUND_PASTE)
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
        # Hide overlay after brief delay to show success
        if self.overlay:
            if pasted:
                # Brief delay to let user see success state
                time.sleep(0.3)
            self.overlay.hide()

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
        # Create tkinter root for overlay (main thread)
        self.tk_root = tk.Tk()
        self.tk_root.withdraw()

        # Create overlay orb
        self.overlay = OverlayOrb()

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

        # Run tray in background thread so tkinter can use main thread
        threading.Thread(target=lambda: self.tray.run(on_tray_ready), daemon=True).start()

        # Run tkinter main loop on main thread
        def tk_mainloop():
            while self.running:
                try:
                    # Process overlay commands from other threads
                    if self.overlay:
                        self.overlay.process_commands()
                    self.tk_root.update()
                    time.sleep(0.01)  # 100fps update rate
                except tk.TclError:
                    break
        tk_mainloop()

if __name__ == "__main__":
    save_config(load_config())  # Ensure config exists
    WinVoice().run()
