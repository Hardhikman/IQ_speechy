import sys
import os

# Set environment variables for phonemizer to find espeak-ng on Windows
os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = 'C:\\Program Files\\eSpeak NG\\libespeak-ng.dll'
os.environ['PHONEMIZER_ESPEAK_PATH'] = 'C:\\Program Files\\eSpeak NG'

def main():
    # Add the neutts-air directory to the Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    neutts_air_path = os.path.join(project_root, 'neutts-air')
    sys.path.insert(0, neutts_air_path)
    
    try:
        from neuttsair.neutts import NeuTTSAir
        import soundfile as sf
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Make sure you have installed the required dependencies:")
        print("pip install -r neutts-air/requirements.txt")
        return

    # Create output directory if it doesn't exist
    output_dir = os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize TTS with GGUF model for better performance
    try:
        tts = NeuTTSAir(
            backbone_repo="neuphonic/neutts-air-q4-gguf",
            backbone_device="cpu",
            codec_repo="neuphonic/neucodec",
            codec_device="cpu"
        )
    except Exception as e:
        print(f"Error initializing NeuTTSAir: {e}")
        return

    # Example text to synthesize
    text = "Hello, this is a test of the NeuTTS Air system. It's capable of voice cloning with just a few seconds of reference audio."

    # Use Dave as the reference speaker
    ref_audio = os.path.join(neutts_air_path, "samples", "dave.wav")
    ref_text_path = os.path.join(neutts_air_path, "samples", "dave.txt")

    # Check if reference files exist
    if not os.path.exists(ref_audio):
        print(f"Reference audio file not found: {ref_audio}")
        return

    if not os.path.exists(ref_text_path):
        print(f"Reference text file not found: {ref_text_path}")
        return

    # Read reference text
    try:
        with open(ref_text_path, "r", encoding="utf-8") as f:
            ref_text = f.read().strip()
    except Exception as e:
        print(f"Error reading reference text: {e}")
        return

    print("Encoding reference audio...")
    try:
        ref_codes = tts.encode_reference(ref_audio)
        print("Generating speech...")
        wav = tts.infer(text, ref_codes, ref_text)
        output_path = os.path.join(output_dir, "test.wav")
        sf.write(output_path, wav, 24000)
        print(f"Audio saved to {output_path}")
    except Exception as e:
        print(f"Error during TTS generation: {e}")

if __name__ == "__main__":
    main()