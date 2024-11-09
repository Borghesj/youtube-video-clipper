import logging
import os
from moviepy.editor import VideoFileClip
import whisper
import subprocess
import json
import time
import torch
import sys

def setup_debug_logging():
    # Ensure we capture all logging levels
    logging.getLogger().setLevel(logging.DEBUG)
    # Add a handler that writes to stderr for immediate output
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)

def validate_video(video_path):
    try:
        with VideoFileClip(video_path) as clip:
            duration = clip.duration
            size = clip.size
            fps = clip.fps
        logging.info(f"Video validated. Duration: {duration:.2f}s, Size: {size}, FPS: {fps}")
        return True
    except Exception as e:
        logging.error(f"Error validating video: {str(e)}")
        return False

def transcribe_audio(audio_path):
    try:
        # Ensure logging is set up properly
        setup_debug_logging()
        
        logging.info("=== Starting Whisper Transcription Process ===")
        logging.info(f"Python version: {sys.version}")
        logging.info(f"Torch version: {torch.__version__}")
        logging.info(f"Whisper version: {whisper.__version__}")
        
        logging.info(f"Processing audio file: {audio_path}")
        logging.info(f"Audio file exists: {os.path.exists(audio_path)}")
        logging.info(f"Audio file size: {os.path.getsize(audio_path)} bytes")
        
        start_time = time.time()
        logging.info(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        logging.info("Checking CUDA availability...")
        cuda_available = torch.cuda.is_available()
        device = "cuda" if cuda_available else "cpu"
        logging.info(f"CUDA available: {cuda_available}")
        logging.info(f"Using device: {device}")

        if device == "cuda":
            logging.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            logging.info(f"CUDA memory allocated: {torch.cuda.memory_allocated(0)}")
            logging.info(f"CUDA memory reserved: {torch.cuda.memory_reserved(0)}")

        logging.info("About to load Whisper model...")
        model = whisper.load_model("base", device=device)
        logging.info("Whisper model loaded successfully.")

        logging.info("Preparing for transcription...")
        logging.info("Setting up transcription options...")
        options = {
            "temperature": 0,
            "best_of": 1,
            "beam_size": 1,
            "verbose": True
        }
        logging.info(f"Transcription options: {options}")
        
        logging.info("Starting actual transcription...")
        result = model.transcribe(
            audio_path,
            **options
        )
        
        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"Transcription completed in {duration:.2f} seconds")
        
        if result and "text" in result:
            logging.info("Successfully extracted text from transcription")
            text_length = len(result["text"])
            logging.info(f"Transcription length: {text_length} characters")
            return result["text"]
        else:
            logging.error("No text found in transcription result")
            return ""

    except Exception as e:
        logging.error("Exception in transcribe_audio:")
        logging.error(f"Error type: {type(e).__name__}")
        logging.error(f"Error message: {str(e)}")
        logging.error("Full traceback:", exc_info=True)
        if torch.cuda.is_available():
            try:
                logging.info(f"Final CUDA memory state: {torch.cuda.memory_allocated(0)}")
                torch.cuda.empty_cache()
                logging.info("CUDA cache cleared")
            except Exception as cuda_e:
                logging.error(f"Error cleaning CUDA memory: {str(cuda_e)}")
        return ""
    finally:
        logging.info("=== Transcription Process Complete ===")

def extract_audio(video_path, audio_path):
    try:
        logging.info(f"Extracting audio from {video_path} to {audio_path}")
        cmd = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'libmp3lame', '-q:a', '2', audio_path]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.info(f"FFmpeg output: {result.stdout}")
        logging.info(f"Audio extracted successfully: {audio_path}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error extracting audio: {e.stderr}")
        return False

def check_audio_file(audio_path):
    if os.path.exists(audio_path):
        file_size = os.path.getsize(audio_path)
        logging.info(f"Audio file size: {file_size} bytes")
        return file_size > 0
    else:
        logging.error(f"Audio file not found: {audio_path}")
        return False

def get_file_info(file_path):
    try:
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', file_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return json.loads(result.stdout)
    except Exception as e:
        logging.error(f"Error getting file info: {str(e)}")
        return None