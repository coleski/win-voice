"""Generate simple sound effects for WinVoice overlay"""
import wave
import struct
import math
import os

SAMPLE_RATE = 44100

def generate_tone(filename, frequency, duration, volume=0.5, fade=True):
    """Generate a simple sine wave tone"""
    n_samples = int(SAMPLE_RATE * duration)
    samples = []

    for i in range(n_samples):
        t = i / SAMPLE_RATE
        # Sine wave
        value = math.sin(2 * math.pi * frequency * t)

        # Apply fade in/out for smooth sound
        if fade:
            fade_samples = int(n_samples * 0.1)
            if i < fade_samples:
                value *= i / fade_samples
            elif i > n_samples - fade_samples:
                value *= (n_samples - i) / fade_samples

        # Scale to 16-bit range
        samples.append(int(value * volume * 32767))

    # Write WAV file
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(struct.pack(f'<{len(samples)}h', *samples))

    print(f"Created: {filename}")

def generate_chirp(filename, freq_start, freq_end, duration, volume=0.5):
    """Generate a frequency sweep (chirp) sound"""
    n_samples = int(SAMPLE_RATE * duration)
    samples = []

    for i in range(n_samples):
        t = i / SAMPLE_RATE
        progress = i / n_samples
        # Linear frequency sweep
        freq = freq_start + (freq_end - freq_start) * progress
        value = math.sin(2 * math.pi * freq * t)

        # Fade envelope
        fade_samples = int(n_samples * 0.15)
        if i < fade_samples:
            value *= i / fade_samples
        elif i > n_samples - fade_samples:
            value *= (n_samples - i) / fade_samples

        samples.append(int(value * volume * 32767))

    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(struct.pack(f'<{len(samples)}h', *samples))

    print(f"Created: {filename}")

def generate_two_tone(filename, freq1, freq2, duration, volume=0.5):
    """Generate two tones in sequence"""
    n_samples = int(SAMPLE_RATE * duration)
    half = n_samples // 2
    samples = []

    for i in range(n_samples):
        t = i / SAMPLE_RATE
        freq = freq1 if i < half else freq2
        value = math.sin(2 * math.pi * freq * t)

        # Fade envelope
        fade_samples = int(n_samples * 0.1)
        if i < fade_samples:
            value *= i / fade_samples
        elif i > n_samples - fade_samples:
            value *= (n_samples - i) / fade_samples

        samples.append(int(value * volume * 32767))

    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(struct.pack(f'<{len(samples)}h', *samples))

    print(f"Created: {filename}")

if __name__ == "__main__":
    sounds_dir = os.path.join(os.path.dirname(__file__), "sounds")
    os.makedirs(sounds_dir, exist_ok=True)

    # Start sound - rising chirp (indicates beginning)
    generate_chirp(
        os.path.join(sounds_dir, "start.wav"),
        freq_start=400, freq_end=800,
        duration=0.15, volume=0.4
    )

    # Stop sound - falling chirp (indicates end)
    generate_chirp(
        os.path.join(sounds_dir, "stop.wav"),
        freq_start=600, freq_end=400,
        duration=0.12, volume=0.35
    )

    # Paste sound - pleasant two-tone chime (success)
    generate_two_tone(
        os.path.join(sounds_dir, "paste.wav"),
        freq1=523, freq2=659,  # C5 to E5
        duration=0.2, volume=0.4
    )

    print("\nSound files generated in:", sounds_dir)
