import sys
import os
import soundfile as sf
import gradio as gr
import numpy as np

# Set environment variables for phonemizer to find espeak-ng on Windows
os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = 'C:\\Program Files\\eSpeak NG\\libespeak-ng.dll'
os.environ['PHONEMIZER_ESPEAK_PATH'] = 'C:\\Program Files\\eSpeak NG'

# Add the neutts-air directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
neutts_air_path = os.path.join(project_root, 'neutts-air')
supertonic_py_path = os.path.join(project_root, 'supertonic', 'py')
sys.path.insert(0, neutts_air_path)
sys.path.insert(0, supertonic_py_path)

# NeuTTS Air Setup
from neuttsair.neutts import NeuTTSAir

tts = None

def initialize_neutts():
    global tts
    if tts is None:
        tts = NeuTTSAir(
            backbone_repo="neuphonic/neutts-air-q4-gguf",
            backbone_device="cpu",
            codec_repo="neuphonic/neucodec",
            codec_device="cpu"
        )
    return tts

def get_available_speakers():
    """Scan samples directory and return list of available speakers."""
    samples_dir = os.path.join(neutts_air_path, "samples")
    speakers = []
    
    if os.path.exists(samples_dir):
        for file in os.listdir(samples_dir):
            if file.endswith('.wav'):
                speaker_name = os.path.splitext(file)[0]
                # Check if corresponding .txt file exists
                txt_file = os.path.join(samples_dir, f"{speaker_name}.txt")
                if os.path.exists(txt_file):
                    # Capitalize first letter for display
                    speakers.append(speaker_name.capitalize())
    
    return sorted(speakers)

def load_reference_speaker(speaker_name):
    """Load pre-defined speakers dynamically from samples folder."""
    samples_dir = os.path.join(neutts_air_path, "samples")
    
    # Convert to lowercase for file lookup
    speaker_lower = speaker_name.lower()
    audio_path = os.path.join(samples_dir, f"{speaker_lower}.wav")
    text_path = os.path.join(samples_dir, f"{speaker_lower}.txt")
    
    if os.path.exists(audio_path) and os.path.exists(text_path):
        with open(text_path, "r", encoding="utf-8") as f:
            ref_text = f.read().strip()
        return audio_path, ref_text
    return None, ""

def chunk_text_for_neutts(text, max_chars=200):
    """Split text into chunks by character limit, preserving complete sentences when possible."""
    import re
    text = text.strip()
    if not text:
        return [text]
    
    # If text is short enough, return as-is
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    remaining = text
    
    while remaining:
        remaining = remaining.strip()
        if not remaining:
            break
            
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break
        
        # Try to find a sentence boundary within the limit
        chunk_candidate = remaining[:max_chars]
        
        # Look for last sentence ending (. ! ?) within the chunk
        last_sentence_end = -1
        for match in re.finditer(r'[.!?]\s+', chunk_candidate):
            last_sentence_end = match.end()
        
        if last_sentence_end > 50:  # Only use if we found a reasonable break point
            chunk = remaining[:last_sentence_end].strip()
            remaining = remaining[last_sentence_end:]
        else:
            # No good sentence break, try to break at last space
            last_space = chunk_candidate.rfind(' ')
            if last_space > 50:
                chunk = remaining[:last_space].strip()
                remaining = remaining[last_space:]
            else:
                # No space found, just cut at max_chars
                chunk = remaining[:max_chars].strip()
                remaining = remaining[max_chars:]
        
        if chunk:
            chunks.append(chunk)
    
    # Verify no text lost
    total_original = len(text.replace(' ', ''))
    total_chunks = sum(len(c.replace(' ', '')) for c in chunks)
    if total_chunks < total_original - 5:
        print(f"WARNING: May have lost text. Original: {total_original} chars, Chunks: {total_chunks} chars")
    
    print(f"Chunked text into {len(chunks)} parts (max {max_chars} chars each)")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {len(chunk)} chars")
    
    return chunks if chunks else [text]

def generate_neutts_speech(text, reference_audio, reference_text, preset_speaker):
    try:
        tts = initialize_neutts()
        
        if preset_speaker != "Custom":
            reference_audio, reference_text = load_reference_speaker(preset_speaker)
        
        if not reference_audio or not reference_text:
            raise ValueError("Reference audio and text are required")
        
        if not text.strip():
            raise ValueError("Text to synthesize is required")
        
        ref_codes = tts.encode_reference(reference_audio)
        
        # Auto-chunk long text (200 chars per chunk to stay within NeuTTS limits)
        chunks = chunk_text_for_neutts(text.strip(), max_chars=200)
        
        # Debug: show each chunk
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}: '{chunk}'")
        
        if len(chunks) == 1:
            # Short text - single generation
            wav = tts.infer(chunks[0], ref_codes, reference_text)
        else:
            # Long text - generate each chunk and concatenate
            audio_segments = []
            silence = np.zeros(int(0.3 * 24000), dtype=np.float32)  # 0.3s pause
            
            for i, chunk in enumerate(chunks):
                print(f"Generating chunk {i+1}/{len(chunks)}...")
                chunk_wav = tts.infer(chunk, ref_codes, reference_text)
                audio_segments.append(chunk_wav)
                if i < len(chunks) - 1:
                    audio_segments.append(silence)
            
            wav = np.concatenate(audio_segments)
        
        output_path = os.path.join(project_root, "output", "neutts_output.wav")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, wav, 24000)
        
        return output_path
    except Exception as e:
        raise gr.Error(f"Error generating speech: {str(e)}")

# Supertonic Setup
from helper import load_text_to_speech, load_voice_style
from scipy.io import wavfile

supertonic_tts = None

def initialize_supertonic():
    global supertonic_tts
    if supertonic_tts is None:
        onnx_dir = os.path.join(project_root, "assets", "onnx")
        supertonic_tts = load_text_to_speech(onnx_dir, use_gpu=False)
    return supertonic_tts

def generate_supertonic_speech(text, voice_style, speed, quality_steps):
    try:
        if not text.strip():
            raise ValueError("Text to synthesize is required")
        
        tts = initialize_supertonic()
        
        # Load voice style
        voice_style_path = os.path.join(project_root, "assets", "voice_styles", f"{voice_style}.json")
        style = load_voice_style([voice_style_path])
        
        # Generate speech
        wav, duration = tts(text.strip(), style, total_step=int(quality_steps), speed=speed)
        
        # Get sample rate from config
        sample_rate = tts.sample_rate
        
        # Save output
        output_path = os.path.join(project_root, "output", "supertonic_output.wav")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert to int16 and save
        wav_int16 = (wav[0] * 32767).astype(np.int16)
        wavfile.write(output_path, sample_rate, wav_int16)
        
        return output_path
    except Exception as e:
        raise gr.Error(f"Error generating speech: {str(e)}")

# Gradio Interface
with gr.Blocks(title="IQ Speechy - Multi-TTS") as demo:
    gr.Markdown("# 🗣️ IQ Speechy - Multi-TTS System")
    
    # Tab 1: NeuTTS Air
    with gr.Tab("NeuTTS Air (Voice Cloning)"):
        gr.Markdown("## Clone voices using reference audio!")
        
        with gr.Row():
            with gr.Column():
                neutts_preset = gr.Dropdown(
                    choices=["Custom"] + get_available_speakers(),
                    value=get_available_speakers()[0] if get_available_speakers() else "Custom",
                    label="Select Preset Speaker"
                )
                
                neutts_ref_audio = gr.Audio(
                    label="Reference Audio (Custom)", 
                    type="filepath",
                    sources=["upload"]
                )
                neutts_ref_text = gr.Textbox(
                    label="Reference Text (Custom)", 
                    placeholder="Enter the exact text spoken in the reference audio...",
                    lines=3
                )
                
                neutts_text = gr.Textbox(
                    label="Text to Synthesize", 
                    placeholder="Enter the text you want to speak in the cloned voice...",
                    lines=3
                )
                neutts_btn = gr.Button("Generate Speech", variant="primary")
            
            with gr.Column():
                neutts_output = gr.Audio(label="Generated Speech", type="filepath")
        
        def update_neutts_fields(preset):
            if preset != "Custom":
                audio, text = load_reference_speaker(preset)
                return gr.Audio(interactive=False), gr.Textbox(value=text, interactive=False)
            else:
                return gr.Audio(interactive=True), gr.Textbox(interactive=True, value="")
        
        neutts_preset.change(
            fn=update_neutts_fields,
            inputs=neutts_preset,
            outputs=[neutts_ref_audio, neutts_ref_text]
        )
        
        neutts_btn.click(
            fn=generate_neutts_speech,
            inputs=[neutts_text, neutts_ref_audio, neutts_ref_text, neutts_preset],
            outputs=neutts_output
        )
        
        gr.Markdown("""
        ### How to use:
        1. Choose a preset speaker (Dave/Jo) or select "Custom" to upload your own
        2. For custom voices, upload a reference audio file and enter its exact text
        3. Enter the new text you want to generate in that voice
        4. Click "Generate Speech"
        
        ⚠️ **Note**: NeuTTS may have issues on Windows due to dependency conflicts.
        """)
    
    # Tab 2: Supertonic
    with gr.Tab("Supertonic (Fast TTS)"):
        gr.Markdown("## ⚡ Lightning Fast On-Device TTS")
        gr.Markdown("167x faster than real-time • 66M parameters • CPU optimized")
        
        with gr.Row():
            with gr.Column():
                supertonic_voice = gr.Dropdown(
                    choices=["M1", "M2", "F1", "F2"],
                    value="M1",
                    label="Voice Style",
                    info="M1/M2 = Male, F1/F2 = Female"
                )
                
                supertonic_speed = gr.Slider(
                    minimum=0.9,
                    maximum=1.5,
                    value=1.05,
                    step=0.05,
                    label="Speech Speed",
                    info="1.0 = Normal, <1.0 = Slower, >1.0 = Faster"
                )
                
                supertonic_steps = gr.Slider(
                    minimum=2,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Quality Steps",
                    info="Higher = Better quality, slower"
                )
                
                supertonic_text = gr.Textbox(
                    label="Text to Synthesize", 
                    placeholder="Enter any text... Supports long-form with auto-chunking!",
                    lines=5
                )
                supertonic_btn = gr.Button("Generate Speech", variant="primary")
            
            with gr.Column():
                supertonic_output = gr.Audio(label="Generated Speech", type="filepath")
        
        supertonic_btn.click(
            fn=generate_supertonic_speech,
            inputs=[supertonic_text, supertonic_voice, supertonic_speed, supertonic_steps],
            outputs=supertonic_output
        )
        
        gr.Markdown("""
        ### Features:
        - ⚡ **Ultra Fast**: 167x faster than real-time on modern hardware
        - 📝 **Long-form**: Supports any length text with automatic chunking
        - 🎨 **4 Voices**: 2 male (M1, M2) and 2 female (F1, F2) voices
        - ⚙️ **Configurable**: Adjust speed and quality to your needs
        """)
    
    # Tab 3: Podcast Generator
    with gr.Tab("🎙️ Podcast Generator"):
        gr.Markdown("## Generate Multi-Speaker Podcasts")
        gr.Markdown("Create conversations with up to 4 different voices!")
        
        with gr.Row():
            with gr.Column():
                podcast_script = gr.Textbox(
                    label="Podcast Script",
                    placeholder="""Write your script using speaker tags:

[M1] Welcome to our podcast! Today we're talking about AI.
[F1] Thanks for having me. It's an exciting topic.
[M1] So what are your thoughts on voice synthesis?
[F1] Well, it's getting incredibly realistic these days!
[M2] I agree, the technology has come a long way.
[F2] And it's only going to get better from here.""",
                    lines=12
                )
                
                with gr.Row():
                    podcast_speed = gr.Slider(
                        minimum=0.9,
                        maximum=1.5,
                        value=1.0,
                        step=0.05,
                        label="Speech Speed"
                    )
                    podcast_steps = gr.Slider(
                        minimum=2,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Quality Steps"
                    )
                
                podcast_pause = gr.Slider(
                    minimum=0.3,
                    maximum=2.0,
                    value=0.5,
                    step=0.1,
                    label="Pause Between Speakers (seconds)"
                )
                
                podcast_btn = gr.Button("🎙️ Generate Podcast", variant="primary")
            
            with gr.Column():
                podcast_output = gr.Audio(label="Generated Podcast", type="filepath")
        
        def generate_podcast(script, speed, quality_steps, pause_duration):
            import re
            try:
                if not script.strip():
                    raise ValueError("Please enter a podcast script")
                
                tts = initialize_supertonic()
                sample_rate = tts.sample_rate
                
                # Parse script for speaker tags
                pattern = r'\[(M1|M2|F1|F2)\]\s*(.+?)(?=\[(?:M1|M2|F1|F2)\]|$)'
                matches = re.findall(pattern, script, re.DOTALL)
                
                if not matches:
                    raise ValueError("No valid speaker tags found. Use [M1], [M2], [F1], or [F2]")
                
                # Generate audio for each segment
                audio_segments = []
                silence = np.zeros(int(pause_duration * sample_rate), dtype=np.float32)
                
                for i, (speaker, text) in enumerate(matches):
                    text = text.strip()
                    if not text:
                        continue
                    
                    # Load voice style
                    voice_style_path = os.path.join(project_root, "assets", "voice_styles", f"{speaker}.json")
                    style = load_voice_style([voice_style_path])
                    
                    # Generate speech
                    wav, _ = tts(text, style, total_step=int(quality_steps), speed=speed)
                    audio_segments.append(wav[0])
                    
                    # Add pause between speakers (except after last)
                    if i < len(matches) - 1:
                        audio_segments.append(silence)
                
                # Concatenate all segments
                full_audio = np.concatenate(audio_segments)
                
                # Save output
                output_path = os.path.join(project_root, "output", "podcast_output.wav")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                wav_int16 = (full_audio * 32767).astype(np.int16)
                wavfile.write(output_path, sample_rate, wav_int16)
                
                return output_path
            except Exception as e:
                raise gr.Error(f"Error generating podcast: {str(e)}")
        
        podcast_btn.click(
            fn=generate_podcast,
            inputs=[podcast_script, podcast_speed, podcast_steps, podcast_pause],
            outputs=podcast_output
        )
        
        gr.Markdown("""
        ### How to use:
        1. Write your script using speaker tags: `[M1]`, `[M2]`, `[F1]`, `[F2]`
        2. Each tag starts a new speaker's dialogue
        3. Adjust speed, quality, and pause duration
        4. Click "Generate Podcast"
        
        ### Voice Guide:
        | Tag | Voice |
        |-----|-------|
        | `[M1]` | Male Voice 1 |
        | `[M2]` | Male Voice 2 |
        | `[F1]` | Female Voice 1 |
        | `[F2]` | Female Voice 2 |
        """)
    
    # Tab 4: NeuTTS Podcast Generator
    with gr.Tab("🎙️ NeuTTS Podcast"):
        gr.Markdown("## Generate Podcasts with Voice Cloning")
        available_voices = get_available_speakers()
        gr.Markdown(f"Available voices: **{', '.join(available_voices)}**")
        
        with gr.Row():
            with gr.Column():
                neutts_podcast_script = gr.Textbox(
                    label="Podcast Script",
                    placeholder="""Write your script using speaker tags:

[Dave] Welcome to the show! Today we have a special guest.
[Jo] Thanks for having me, it's great to be here.
[Dave] So tell us about your latest project.
[Jo] Well, it's been quite an adventure...""",
                    lines=12
                )
                
                neutts_podcast_pause = gr.Slider(
                    minimum=0.3,
                    maximum=2.0,
                    value=0.5,
                    step=0.1,
                    label="Pause Between Speakers (seconds)"
                )
                
                neutts_podcast_btn = gr.Button("🎙️ Generate NeuTTS Podcast", variant="primary")
            
            with gr.Column():
                neutts_podcast_output = gr.Audio(label="Generated Podcast", type="filepath")
        
        def generate_neutts_podcast(script, pause_duration):
            import re
            try:
                if not script.strip():
                    raise ValueError("Please enter a podcast script")
                
                tts = initialize_neutts()
                
                # Get available speakers dynamically
                available_speakers = get_available_speakers()
                if not available_speakers:
                    raise ValueError("No speaker samples found in neutts-air/samples/")
                
                # Build dynamic regex pattern for all available speakers
                speaker_pattern = '|'.join(available_speakers)
                pattern = rf'\[({speaker_pattern})\]\s*(.+?)(?=\[(?:{speaker_pattern})\]|$)'
                matches = re.findall(pattern, script, re.DOTALL | re.IGNORECASE)
                
                if not matches:
                    raise ValueError(f"No valid speaker tags found. Use: {', '.join([f'[{s}]' for s in available_speakers])}")
                
                # Pre-load reference codes for used speakers
                speaker_codes = {}
                speaker_texts = {}
                used_speakers = set(m[0].capitalize() for m in matches)
                
                for speaker in used_speakers:
                    audio_path, ref_text = load_reference_speaker(speaker)
                    if audio_path:
                        speaker_codes[speaker] = tts.encode_reference(audio_path)
                        speaker_texts[speaker] = ref_text
                
                # Generate audio for each segment
                audio_segments = []
                silence = np.zeros(int(pause_duration * 24000), dtype=np.float32)
                
                for i, (speaker, text) in enumerate(matches):
                    text = text.strip()
                    if not text:
                        continue
                    
                    speaker = speaker.capitalize()  # Normalize speaker name
                    if speaker not in speaker_codes:
                        raise ValueError(f"Unknown speaker: {speaker}")
                    
                    # Generate speech with chunking
                    chunks = chunk_text_for_neutts(text, max_words=100)
                    for chunk in chunks:
                        wav = tts.infer(chunk, speaker_codes[speaker], speaker_texts[speaker])
                        audio_segments.append(wav)
                    
                    # Add pause between speakers (except after last)
                    if i < len(matches) - 1:
                        audio_segments.append(silence)
                
                # Concatenate all segments
                full_audio = np.concatenate(audio_segments)
                
                # Save output
                output_path = os.path.join(project_root, "output", "neutts_podcast_output.wav")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                sf.write(output_path, full_audio, 24000)
                
                return output_path
            except Exception as e:
                raise gr.Error(f"Error generating podcast: {str(e)}")
        
        neutts_podcast_btn.click(
            fn=generate_neutts_podcast,
            inputs=[neutts_podcast_script, neutts_podcast_pause],
            outputs=neutts_podcast_output
        )
        
        # Generate dynamic help text
        voice_tags = ', '.join([f'`[{v}]`' for v in available_voices])
        voice_table = '\n'.join([f'        | `[{v}]` | {v} voice |' for v in available_voices])
        
        gr.Markdown(f"""
        ### How to use:
        1. Write your script using speaker tags: {voice_tags}
        2. Each tag starts a new speaker's dialogue
        3. Adjust pause duration between speakers
        4. Click "Generate NeuTTS Podcast"
        
        ### Voice Guide:
        | Tag | Voice |
        |-----|-------|
{voice_table}
        
        ⚠️ **Note**: NeuTTS is slower than Supertonic but provides voice cloning.
        💡 **Tip**: Add new voices by placing `.wav` and `.txt` files in `neutts-air/samples/`
        """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)