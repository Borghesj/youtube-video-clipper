import logging
import os
import json
import traceback
import atexit
from video_processing.downloader import download_video
from video_processing.clip_creator import create_clips
from utils.logging_config import setup_logging
from utils.file_operations import safe_remove
from ai_enhancement.engaging_moments import identify_engaging_moments
from utils.video_utils import validate_video, transcribe_audio, extract_audio, check_audio_file, get_file_info
from config import OUTPUT_DIRECTORY

def cleanup_temp_files(temp_files):
    """Clean up temporary files"""
    for file_path in temp_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logging.error(f"Error cleaning up {file_path}: {str(e)}")

def process_video(video_url):
    setup_logging()
    output_path = os.path.join(OUTPUT_DIRECTORY, "downloaded_video.mp4")
    audio_path = os.path.join(OUTPUT_DIRECTORY, "extracted_audio.mp3")
    transcript_path = os.path.join(OUTPUT_DIRECTORY, "full_transcript.txt")
    
    # List of temporary files to clean up
    temp_files = [output_path, audio_path, transcript_path]
    
    try:
        logging.info(f"Processing video from URL: {video_url}")
        
        logging.info("Step 1: Downloading video...")
        video_name = download_video(video_url, output_path)
        if not video_name:
            logging.error("Failed to download video. Aborting process.")
            return []
        logging.info(f"Video downloaded: {video_name}")
        
        logging.info("Step 2: Validating video...")
        if not validate_video(output_path):
            logging.error("Downloaded video is invalid. Aborting process.")
            return []
        logging.info("Video validated successfully.")
        
        logging.info("Step 3: Extracting audio...")
        if extract_audio(output_path, audio_path):
            if check_audio_file(audio_path):
                logging.info("Audio extracted and verified successfully.")
            else:
                logging.error("Extracted audio file is invalid or empty.")
                return []
        else:
            logging.error("Failed to extract audio.")
            return []
        
        logging.info("Step 4: Getting video file info...")
        video_info = get_file_info(output_path)
        logging.info(f"Video file info:\n{json.dumps(video_info, indent=2)}")

        logging.info("Step 5: Getting audio file info...")
        audio_info = get_file_info(audio_path)
        logging.info(f"Audio file info:\n{json.dumps(audio_info, indent=2)}")
        
        logging.info("Step 6: Transcribing audio...")
        transcript = transcribe_audio(audio_path)
        if not transcript:
            logging.error("Failed to transcribe audio. Aborting process.")
            return []
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(transcript)
        logging.info(f"Transcription completed. Transcript saved to {transcript_path}")
        logging.info(f"Transcript preview (first 200 characters): {transcript[:200]}...")
        
        logging.info("Step 7: Identifying engaging moments...")
        engaging_moments = identify_engaging_moments(output_path, transcript, video_name)
        if not engaging_moments:
            logging.warning("No engaging moments identified.")
            return []
        logging.info(f"Identified {len(engaging_moments)} engaging moments.")
        
        logging.info("Step 8: Creating clips...")
        clips = create_clips(output_path, engaging_moments)
        logging.info(f"Number of clips created: {len(clips)}")
        
        return clips
    except Exception as e:
        logging.error(f"An error occurred during video processing: {str(e)}")
        logging.error(traceback.format_exc())
        return []
    finally:
        # Clean up temporary files in the finally block
        cleanup_temp_files(temp_files)

def main():
    try:
        video_url = input("Enter the YouTube URL: ")
        processed_clips = process_video(video_url)
        if processed_clips:
            logging.info("Video processing completed successfully.")
            for i, clip in enumerate(processed_clips, 1):
                logging.info(f"Clip {i}:")
                logging.info(f"  File: {clip['file']}")
                logging.info(f"  Title: {clip['title']}")
                logging.info(f"  Description: {clip['description']}")
                logging.info(f"  Text: {clip['text']}")
                logging.info("---")
        else:
            logging.warning("No clips were processed successfully.")
    except Exception as e:
        logging.error(f"An unexpected error occurred in main: {str(e)}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()