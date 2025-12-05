# IQ Speechy

A multi-engine text-to-speech (TTS) system with voice cloning, fast preset voices, and podcast generation.

## Features

### 🎙️ NeuTTS Air (Voice Cloning)
- Clone any voice with 3-6 seconds of reference audio
- Preset speakers: Dave, Jo
- Custom voice upload support
- **Auto-chunking** for long-form text (200 chars per chunk)

### ⚡ Supertonic (Fast TTS)
- **167x faster** than real-time
- 4 preset voices: M1, M2 (male), F1, F2 (female)
- Adjustable speed (0.9x - 1.5x) and quality steps
- Long-form text with auto-chunking
- CPU optimized (no GPU required)

### 🎙️ Podcast Generators

**Supertonic Podcast** - Fast multi-speaker:
```
[M1] Welcome to our podcast!
[F1] Thanks for having me.
[M2] Great to be here.
[F2] Let's dive in!
```

**NeuTTS Podcast** - Voice cloned speakers:
```
[Dave] Welcome to the show!
[Jo] Thanks for having me.
```

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Hardhikman/IQ_speechy.git
cd neutts-tts-project
```

### 2. Install dependencies
```bash
python -m venv neutts-env
neutts-env\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 3. Clone Supertonic repo and assets
```bash
git clone https://github.com/supertone-inc/supertonic.git supertonic
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
├── app.py              # Main Gradio application (4 tabs)
├── requirements.txt    # Python dependencies
├── assets/             # Supertonic ONNX models & voice styles
├── neutts-air/         # NeuTTS Air library
├── supertonic/py/      # Supertonic Python inference code
└── output/             # Generated audio files
```

## Tabs

| Tab | Engine | Best For |
|-----|--------|----------|
| **NeuTTS Air** | Voice cloning | Custom voice replication |
| **Supertonic** | Preset voices | Fast single-voice TTS |
| **🎙️ Podcast Generator** | Supertonic | Fast 4-voice podcasts |
| **🎙️ NeuTTS Podcast** | NeuTTS | Voice-cloned 2-speaker podcasts |

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