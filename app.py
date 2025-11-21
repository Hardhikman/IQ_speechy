import sys
import os
import soundfile as sf
import gradio as gr
import numpy as np
import threading
import time

# Set environment variables for phonemizer to find espeak-ng on Windows
os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = 'C:\\Program Files\\eSpeak NG\\libespeak-ng.dll'
os.environ['PHONEMIZER_ESPEAK_PATH'] = 'C:\\Program Files\\eSpeak NG'

# Add the neutts-air directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
neutts_air_path = os.path.join(project_root, 'neutts-air')
sys.path.insert(0, neutts_air_path)

from neuttsair.neutts import NeuTTSAir

# Initialize TTS model
tts = None

def initialize_tts():
    global tts
    if tts is None:
        tts = NeuTTSAir(
            backbone_repo="neuphonic/neutts-air-q4-gguf",
            backbone_device="cpu",
            codec_repo="neuphonic/neucodec",
            codec_device="cpu"
        )
    return tts

def load_reference_speaker(speaker_name):
    """Load pre-defined speakers (Dave or Jo)"""
    speaker_files = {
        "Dave": {
            "audio": os.path.join(neutts_air_path, "samples", "dave.wav"),
            "text": os.path.join(neutts_air_path, "samples", "dave.txt")
        },
        "Jo": {
            "audio": os.path.join(neutts_air_path, "samples", "jo.wav"),
            "text": os.path.join(neutts_air_path, "samples", "jo.txt")
        }
    }
    
    if speaker_name in speaker_files:
        speaker = speaker_files[speaker_name]
        with open(speaker["text"], "r", encoding="utf-8") as f:
            ref_text = f.read().strip()
        return speaker["audio"], ref_text
    return None, ""

def generate_speech(text, reference_audio, reference_text, preset_speaker):
    try:
        # Initialize TTS if not already done
        tts = initialize_tts()
        
        # Use preset speaker if selected
        if preset_speaker != "Custom":
            reference_audio, reference_text = load_reference_speaker(preset_speaker)
        
        # Validate inputs
        if not reference_audio or not reference_text:
            raise ValueError("Reference audio and text are required")
        
        if not text.strip():
            raise ValueError("Text to synthesize is required")
        
        # Encode reference audio
        ref_codes = tts.encode_reference(reference_audio)
        
        # Generate speech
        wav = tts.infer(text.strip(), ref_codes, reference_text)
        
        # Save to temporary file
        output_path = "temp_output.wav"
        sf.write(output_path, wav, 24000)
        
        return output_path
    except Exception as e:
        raise gr.Error(f"Error generating speech: {str(e)}")

# Gradio interface
with gr.Blocks(title="NeuTTS Voice Cloning") as demo:
    gr.Markdown("# 🗣️ NeuTTS Voice Cloning")
    
    with gr.Tab("Text to Speech"):
        gr.Markdown("## Clone voices and generate speech using reference audio!")
        
        with gr.Row():
            with gr.Column():
                preset_speaker = gr.Dropdown(
                    choices=["Custom", "Dave", "Jo"],
                    value="Dave",
                    label="Select Preset Speaker"
                )
                
                reference_audio = gr.Audio(
                    label="Reference Audio (Custom)", 
                    type="filepath",
                    sources=["upload"]
                )
                reference_text = gr.Textbox(
                    label="Reference Text (Custom)", 
                    placeholder="Enter the exact text spoken in the reference audio...",
                    lines=3
                )
                
                text_input = gr.Textbox(
                    label="Text to Synthesize", 
                    placeholder="Enter the text you want to speak in the cloned voice...",
                    lines=3
                )
                generate_btn = gr.Button("Generate Speech", variant="primary")
            
            with gr.Column():
                output_audio = gr.Audio(label="Generated Speech", type="filepath")
        
        def update_reference_fields(preset):
            if preset != "Custom":
                audio, text = load_reference_speaker(preset)
                return gr.Audio(interactive=False), gr.Textbox(value=text, interactive=False)
            else:
                return gr.Audio(interactive=True), gr.Textbox(interactive=True, value="")
        
        preset_speaker.change(
            fn=update_reference_fields,
            inputs=preset_speaker,
            outputs=[reference_audio, reference_text]
        )
        
        generate_btn.click(
            fn=generate_speech,
            inputs=[text_input, reference_audio, reference_text, preset_speaker],
            outputs=output_audio
        )
        
        gr.Markdown("""
        ### How to use:
        1. Choose a preset speaker (Dave/Jo) or select "Custom" to upload your own
        2. For custom voices, upload a reference audio file and enter its exact text
        3. Enter the new text you want to generate in that voice
        4. Click "Generate Speech"
        """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)