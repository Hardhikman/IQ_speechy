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

# Initialize STT model (Whisper)
stt_pipeline = None

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

def initialize_stt():
    global stt_pipeline
    if stt_pipeline is None:
        from transformers import pipeline
        stt_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small", ignore_warning=True)
    return stt_pipeline

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

def speech_to_text(audio):
    try:
        if audio is None:
            return "No audio provided"
        
        # Initialize STT if not already done
        stt = initialize_stt()
        
        # Convert Gradio audio format to what Whisper expects
        sampling_rate, data = audio
        
        # If stereo, convert to mono
        if len(data.shape) > 1:
            data = data.mean(axis=1)
            
        # Normalize audio data
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
            
        # Transcribe audio (explicitly set language to English to avoid ambiguity)
        # Increase max_new_tokens for longer transcriptions and return_timestamps for better handling
        result = stt(
            {"sampling_rate": sampling_rate, "raw": data}, 
            generate_kwargs={
                "language": "en", 
                "task": "transcribe",
                "max_new_tokens": 448,  # Increased for longer transcriptions
                "return_timestamps": False  # Disable timestamps for cleaner output
            }
        )
        # Handle different possible return formats
        if isinstance(result, dict):
            return result.get("text", str(result))
        else:
            return str(result)
    except Exception as e:
        raise gr.Error(f"Error transcribing speech: {str(e)}")

# For live streaming functionality
class StreamTranscriber:
    def __init__(self):
        self.transcriber = None
        self.is_recording = False
        self.audio_buffer = []
        self.transcription = ""
        
    def start_streaming(self):
        self.transcriber = initialize_stt()
        self.is_recording = True
        self.audio_buffer = []
        self.transcription = ""
        return "Live streaming started. Speak into your microphone."
    
    def stop_streaming(self):
        self.is_recording = False
        return self.transcription
    
    def process_audio_chunk(self, audio_chunk):
        if not self.is_recording or audio_chunk is None:
            return self.transcription
            
        try:
            # Convert Gradio audio format
            sampling_rate, data = audio_chunk
            
            # If stereo, convert to mono
            if len(data.shape) > 1:
                data = data.mean(axis=1)
                
            # Normalize audio data
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
                
            # Buffer the audio data
            self.audio_buffer.extend(data)
            
            # Process in chunks of appropriate size (approximately 5 seconds at 16kHz)
            if len(self.audio_buffer) >= 16000 * 5:
                # Convert to numpy array
                audio_data = np.array(self.audio_buffer)
                
                # Transcribe the buffered audio
                if self.transcriber is not None:
                    result = self.transcriber(
                        {"sampling_rate": sampling_rate, "raw": audio_data}, 
                        generate_kwargs={
                            "language": "en", 
                            "task": "transcribe"
                        }
                    )
                else:
                    result = {"text": "Transcriber not initialized"}
                
                # Update transcription
                if isinstance(result, dict):
                    new_text = result.get("text", "")
                    if new_text.strip():
                        self.transcription += " " + new_text.strip()
                
                # Clear buffer
                self.audio_buffer = []
                
            return self.transcription
        except Exception as e:
            return f"Error in streaming: {str(e)}"

# Global stream transcriber instance
stream_transcriber = StreamTranscriber()

def start_live_streaming():
    return stream_transcriber.start_streaming()

def stop_live_streaming():
    return stream_transcriber.stop_streaming()

def process_live_audio(audio_chunk):
    return stream_transcriber.process_audio_chunk(audio_chunk)

# Gradio interface
with gr.Blocks(title="NeuTTS Voice Cloning") as demo:
    gr.Markdown("# 🗣️ NeuTTS Voice Cloning & Speech Recognition")
    
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
    
    with gr.Tab("Speech to Text"):
        gr.Markdown("## Transcribe audio to text using Whisper")
        
        with gr.Row():
            with gr.Column():
                stt_audio = gr.Audio(label="Upload Audio for Transcription", type="numpy")
                stt_button = gr.Button("Transcribe", variant="primary")
            
            with gr.Column():
                stt_output = gr.Textbox(label="Transcription", lines=5)
        
        stt_button.click(
            fn=speech_to_text,
            inputs=stt_audio,
            outputs=stt_output
        )
        
        gr.Markdown("""
        ### How to use:
        1. Upload an audio file (WAV, MP3, etc.)
        2. Click "Transcribe" to convert speech to text
        3. The transcribed text will appear in the output box
        """)
    
    with gr.Tab("Live Streaming"):
        gr.Markdown("## Live speech-to-text streaming")
        
        with gr.Row():
            with gr.Column():
                live_audio = gr.Audio(
                    label="Live Audio Input", 
                    type="numpy", 
                    sources=["microphone"],
                    streaming=True
                )
                control_row = gr.Row()
                with control_row:
                    start_button = gr.Button("Start Streaming")
                    stop_button = gr.Button("Stop Streaming")
            
            with gr.Column():
                live_output = gr.Textbox(label="Live Transcription", lines=10)
        
        start_button.click(
            fn=start_live_streaming,
            outputs=live_output
        )
        
        stop_button.click(
            fn=stop_live_streaming,
            outputs=live_output
        )
        
        live_audio.stream(
            fn=process_live_audio,
            inputs=live_audio,
            outputs=live_output,
            time_limit=900  # 15 minutes max
        )
        
        gr.Markdown("""
        ### How to use:
        1. Click "Start Streaming" to begin live transcription
        2. Speak into your microphone
        3. The transcribed text will appear in real-time
        4. Click "Stop Streaming" when finished
        """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)