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
from config import OUTPUT_DIRECTORY, OPENAI_API_KEY
import openai

# Set up OpenAI API
openai.api_key = OPENAI_API_KEY

def generate_tiktok_description(video_name: str, moment_text: str) -> dict:
    """Generate TikTok-optimized description and hashtags for a video clip"""
    
    prompt = f"""Generate a TikTok description and hashtags for this video clip.
Context: This is a clip from "{video_name}"
Clip content: {moment_text}

Format the response as a JSON object with these exact keys:
- description: A short, engaging description (max 150 characters)
- hashtags: A list of 4-6 relevant hashtags (including the # symbol)
- hook: A one-line attention grabber (optional)

Make the description conversational and engaging. Include emojis where appropriate.
Focus on creating viral, shareable content while maintaining authenticity.
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a TikTok content optimization expert who creates engaging, viral descriptions and hashtags."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        # Parse the JSON response
        description_data = json.loads(response.choices[0].message['content'])
        
        # Format hashtags as a single string
        hashtags_str = ' '.join(description_data['hashtags'])
        
        # Combine hook (if present) with description
        final_description = description_data.get('hook', '') + '\n\n' + description_data['description']
        
        return {
            'description': final_description.strip(),
            'hashtags': hashtags_str,
            'full_text': f"{final_description.strip()}\n\n{hashtags_str}"
        }
        
    except Exception as e:
        logging.error(f"Error generating TikTok description: {str(e)}")
        return {
            'description': moment_text[:150] + '...',
            'hashtags': '#fyp #viral',
            'full_text': f"{moment_text[:150]}...\n\n#fyp #viral"
        }

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
        
        logging.info("Step 8: Creating clips and generating TikTok descriptions...")
        clips = create_clips(output_path, engaging_moments)
        
        # Generate TikTok descriptions for each clip
        for i, clip in enumerate(clips, 1):
            tiktok_content = generate_tiktok_description(video_name, clip['text'])
            clip['tiktok'] = tiktok_content
            
            # Save TikTok content to a text file
            tiktok_text_path = os.path.join(
                OUTPUT_DIRECTORY, 
                f"tiktok_text_{i}_{clip['title'].replace(' ', '_')}.txt"
            )
            with open(tiktok_text_path, 'w', encoding='utf-8') as f:
                f.write(f"TikTok Description:\n{tiktok_content['description']}\n\n")
                f.write(f"Hashtags:\n{tiktok_content['hashtags']}\n\n")
                f.write(f"Full Text:\n{tiktok_content['full_text']}")
            
            logging.info(f"Generated TikTok content for clip {i}")
        
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
                logging.info(f"  TikTok Description: {clip['tiktok']['description']}")
                logging.info(f"  TikTok Hashtags: {clip['tiktok']['hashtags']}")
                logging.info("---")
        else:
            logging.warning("No clips were processed successfully.")
    except Exception as e:
        logging.error(f"An unexpected error occurred in main: {str(e)}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()