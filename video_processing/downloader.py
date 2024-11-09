import logging
import yt_dlp
import subprocess
import json
import os

def get_video_info(video_path):
    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', video_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(result.stdout)

def download_video(video_url, output_path):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': output_path,
        'merge_output_format': 'mp4',
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }],
        'verbose': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            video_name = info.get('title', 'Unknown Video')
        
        logging.info(f"Video downloaded successfully: {output_path}")
        logging.info(f"File size: {os.path.getsize(output_path)} bytes")

        # Check video information using ffprobe
        video_info = get_video_info(output_path)
        logging.info(f"Video info: {json.dumps(video_info, indent=2)}")

        # Check for audio streams
        audio_streams = [stream for stream in video_info.get('streams', []) if stream['codec_type'] == 'audio']
        if audio_streams:
            logging.info(f"Audio streams found: {len(audio_streams)}")
            for stream in audio_streams:
                logging.info(f"Audio codec: {stream.get('codec_name')}, channels: {stream.get('channels')}")
        else:
            logging.warning("No audio streams found in the video file")

        return video_name
    except Exception as e:
        logging.error(f"Error downloading video: {str(e)}", exc_info=True)
        return None