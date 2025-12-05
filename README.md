# IQ Speechy

A multi-engine text-to-speech (TTS) system with voice cloning, fast preset voices, and podcast generation.

## Features

### 🎙️ NeuTTS Air (Voice Cloning)
- Clone any voice with 3-6 seconds of reference audio
- Preset speakers: Dave, Jo
- Custom voice upload support

### ⚡ Supertonic (Fast TTS)
- **167x faster** than real-time
- 4 preset voices: M1, M2 (male), F1, F2 (female)
- Adjustable speed (0.9x - 1.5x)
- Long-form text with auto-chunking
- CPU optimized (no GPU required)

### 🎙️ Podcast Generator
- Create multi-speaker conversations
- Up to 4 different voices
- Script format with speaker tags: `[M1]`, `[M2]`, `[F1]`, `[F2]`
- Adjustable pause between speakers
- Auto-combines into single podcast audio

## Getting Started

### 1. Clone the repository
```bash
git clone <repo-url>
cd neutts-tts-project
```

### 2. Install dependencies
```bash
python -m venv neutts-env
neutts-env\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 3. Download Supertonic assets
```bash
git lfs install
git clone https://huggingface.co/Supertone/supertonic assets
```

### 4. Run the application
```bash
python app.py
```

Open http://localhost:7860 in your browser.

## Project Structure

```
neutts-tts-project/
├── app.py              # Main Gradio application
├── assets/             # Supertonic ONNX models & voice styles
├── neutts-air/         # NeuTTS Air library
├── supertonic/py/      # Supertonic Python inference code
└── output/             # Generated audio files
```

## Usage

The web interface has three tabs:

| Tab | Engine | Best For |
|-----|--------|----------|
| **NeuTTS Air** | Voice cloning | Custom voice replication |
| **Supertonic** | Preset voices | Fast, long-form TTS |
| **Podcast Generator** | Multi-speaker | Conversations, podcasts |

### Podcast Script Example
```
[M1] Welcome to our podcast! Today we're discussing AI.
[F1] Thanks for having me. It's an exciting topic.
[M1] What are your thoughts on voice synthesis?
[F1] It's getting incredibly realistic these days!
```

## Dependencies

- Python 3.11+
- gradio
- onnxruntime
- scipy
- soundfile
- numpy
- torch (for NeuTTS)

## License

See individual component licenses (NeuTTS Air, Supertonic).