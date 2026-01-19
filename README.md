# Win-Voice

Push-to-talk voice dictation for Windows using Faster Whisper AI.

**Hold a key to record, release to transcribe and paste anywhere.**

## Quick Start

1. Download and run `install.bat` (installs Python dependencies)
2. Run `python dictate.py`
3. Configure your hotkey, microphone, and model on first launch
4. Hold your hotkey, speak, release to paste

## Features

- **GPU-accelerated** transcription (NVIDIA CUDA)
- **Settings GUI** - choose hotkey, microphone, model, language
- **Visual overlay** - shows recording/processing status
- **Clipboard paste** - works in any application
- **Auto-downloads** Whisper model on first run

## Requirements

- Windows 10/11
- Python 3.9+
- NVIDIA GPU (optional, falls back to CPU)

## Installation

```bash
git clone https://github.com/coleski/win-voice.git
cd win-voice
install.bat
```

Or manually:
```bash
pip install -r requirements.txt
python dictate.py
```

## Settings

On first run, a settings window appears:

- **Push-to-Talk Key**: Alt, Ctrl, Shift, F1-F8, etc.
- **Model**: tiny (fast), base, small, medium, large-v3 (accurate)
- **Microphone**: Select your input device
- **Language**: en, es, fr, de, auto-detect, etc.

Settings are saved to `config.json`. Delete it to reset.

## Usage

1. Run `python dictate.py` or `run_dictation.bat`
2. Wait for "Ready" overlay (model loading)
3. Hold your hotkey and speak
4. Release to transcribe and paste

### Overlay Colors

- **Yellow**: Loading/Processing
- **Red**: Recording
- **Green**: Success (shows transcribed text)

## Auto-Start with Windows

1. Press `Win+R`, type `shell:startup`
2. Create shortcut to `VoiceDictation.vbs`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No GPU acceleration | Install `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` |
| Slow transcription | Use `tiny` model in settings |
| No paste | Run as administrator |
| Wrong microphone | Change in settings, delete `config.json` |

## License

MIT
