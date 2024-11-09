import logging
import os
from moviepy.editor import VideoFileClip
import cv2
import numpy as np
from config import OUTPUT_DIRECTORY
import whisper
import torch
import tempfile
import subprocess
import json

def extract_audio_segment(video_path, start_time, end_time, temp_dir):
    """Extract audio segment for precise speech recognition"""
    temp_audio = os.path.join(temp_dir, "temp_segment.wav")
    cmd = [
        'ffmpeg', '-i', video_path,
        '-ss', str(start_time),
        '-t', str(end_time - start_time),
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        temp_audio,
        '-y'
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return temp_audio

def get_precise_word_timings(video_path, start_time, end_time, text):
    """Get precise word timings using Whisper's alignment"""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract the specific audio segment
            audio_path = extract_audio_segment(video_path, start_time, end_time, temp_dir)
            
            # Load Whisper model
            model = whisper.load_model("tiny")  # Using tiny model for faster alignment
            
            # Get word-level timestamps
            result = model.transcribe(
                audio_path,
                word_timestamps=True,
                initial_prompt=text  # Use the known text to improve accuracy
            )
            
            # Process word timings
            word_timings = []
            for segment in result["segments"]:
                for word in segment["words"]:
                    word_timings.append({
                        'word': word["word"].strip(),
                        'start': word["start"],
                        'end': word["end"],
                        'highlighted': False
                    })
            
            return word_timings
            
    except Exception as e:
        logging.error(f"Error in word timing extraction: {str(e)}")
        return None

def get_sentence_groups(word_timings):
    """Group words into shorter, more manageable phrases"""
    phrases = []
    current_phrase = []
    word_count = 0
    
    for word in word_timings:
        current_phrase.append(word)
        word_count += 1
        
        # Break phrases at natural points or max length (reduced from 12 to 6)
        if (word['word'].rstrip() in '.!?' or 
            word_count >= 6 or  # Shorter max length
            word['word'].rstrip() in ',;' and word_count >= 4):  # Break at commas if phrase is getting long
            phrases.append(current_phrase)
            current_phrase = []
            word_count = 0
    
    # Add any remaining words
    if current_phrase:
        phrases.append(current_phrase)
    
    return phrases

def create_social_media_text(frame, word_timings, current_time):
    """Create social media style text with better size control"""
    frame_height, frame_width = frame.shape[:2]
    result = frame.copy()
    
    # Adjusted font settings
    font = cv2.FONT_HERSHEY_DUPLEX
    base_size = frame_width / 25.0  # Slightly smaller base size
    font_scale = base_size / 30.0
    thickness = max(int(font_scale * 4), 4)
    
    # Find current word and its phrase
    current_word = None
    current_phrase = None
    phrase_groups = get_sentence_groups(word_timings)
    
    for phrase in phrase_groups:
        for word in phrase:
            if word['start'] <= current_time <= word['end']:
                current_word = word
                current_phrase = phrase
                break
        if current_phrase:
            break
    
    if not current_phrase:
        return result
    
    # Calculate text position
    y_position = int(frame_height * 0.75)
    
    # Build phrase with proper spacing
    text_elements = []
    total_width = 0
    max_width = int(frame_width * 0.85)  # Maximum width constraint
    
    # First pass: calculate total width
    for word in current_phrase:
        size = cv2.getTextSize(word['word'] + ' ', font, font_scale, thickness)[0]
        text_elements.append({
            'word': word['word'],
            'width': size[0],
            'is_current': word == current_word
        })
        total_width += size[0]
    
    # Adjust font scale if text is too wide
    if total_width > max_width:
        scale_factor = max_width / total_width
        font_scale *= scale_factor
        thickness = max(int(font_scale * 4), 3)
        
        # Recalculate widths with new font scale
        total_width = 0
        for element in text_elements:
            size = cv2.getTextSize(element['word'] + ' ', font, font_scale, thickness)[0]
            element['width'] = size[0]
            total_width += size[0]
    
    # Center the text block
    x_position = (frame_width - total_width) // 2
    
    # Create enhanced background with proper boundaries
    padding_x = int(base_size * 0.6)
    padding_y = int(base_size * 0.4)
    bg_height = int(base_size * 1.8)
    bg_y = y_position - int(base_size * 1.3)
    
    # Draw gradient background
    overlay = result.copy()
    bg_rect_points = np.array([
        [x_position - padding_x, bg_y],
        [x_position + total_width + padding_x, bg_y],
        [x_position + total_width + padding_x, bg_y + bg_height + padding_y],
        [x_position - padding_x, bg_y + bg_height + padding_y]
    ], np.int32)
    
    # Create gradient overlay
    cv2.fillPoly(overlay, [bg_rect_points], (0, 0, 0))
    cv2.addWeighted(overlay, 0.7, result, 0.3, 0, result)
    
    # Draw text
    current_x = x_position
    for element in text_elements:
        word = element['word']
        is_current = element['is_current']
        
        # Enhanced outline
        outline_thickness = thickness * 2
        outline_color = (0, 0, 0)
        
        # Multiple outline layers for better visibility
        for offset in [(2,2), (-2,-2), (2,-2), (-2,2), (0,2), (2,0), (0,-2), (-2,0)]:
            cv2.putText(
                result,
                word,
                (current_x + offset[0], y_position + offset[1]),
                font,
                font_scale,
                outline_color,
                outline_thickness,
                cv2.LINE_AA
            )
        
        # Main text
        text_color = (255, 0, 0) if is_current else (255, 255, 255)
        cv2.putText(
            result,
            word,
            (current_x, y_position),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA
        )
        
        # Add space after word
        current_x += element['width'] + int(base_size * 0.1)
    
    return result

def create_clips(video_path, engaging_moments):
    """Create video clips with precisely aligned captions"""
    clips = []
    try:
        logging.info(f"Opening video file: {video_path}")
        with VideoFileClip(video_path) as video:
            if not engaging_moments:
                logging.warning("No engaging moments identified.")
                return []

            for index, moment in enumerate(engaging_moments, 1):
                try:
                    start_time = max(0, min(moment.get('start', 0), video.duration))
                    end_time = min(video.duration, moment.get('end', video.duration))
                    clip_duration = end_time - start_time
                    title = moment.get('title', f"Clip {index}")

                    # Get precise word timings using Whisper
                    word_timings = get_precise_word_timings(
                        video_path,
                        start_time,
                        end_time,
                        moment['text']
                    )
                    
                    if not word_timings:
                        logging.error(f"Failed to get word timings for clip {index}")
                        continue

                    clip = video.subclip(start_time, end_time)
                    
                    # Resize for TikTok/Instagram
                    w, h = clip.size
                    target_aspect = 9/16
                    current_aspect = w/h
                    
                    if current_aspect > target_aspect:
                        new_w = int(h * target_aspect)
                        new_w = new_w - (new_w % 2)  # Ensure even width
                        resized_clip = clip.crop(x_center=w/2, width=new_w, height=h)
                    else:
                        new_h = int(w / target_aspect)
                        new_h = new_h - (new_h % 2)  # Ensure even height
                        resized_clip = clip.crop(y_center=h/2, width=w, height=new_h)
                    
                    def add_captions(get_frame, t):
                        frame = get_frame(t)
                        return create_social_media_text(frame, word_timings, t)
                    
                    final_clip = resized_clip.fl(add_captions)
                    
                    output_filename = f"tiktok_clip_{index}_{title.replace(' ', '_')}.mp4"
                    output_path = os.path.join(OUTPUT_DIRECTORY, output_filename)
                    
                    final_clip.write_videofile(
                        output_path,
                        codec='libx264',
                        audio_codec='aac',
                        temp_audiofile=os.path.join(OUTPUT_DIRECTORY, f"temp-audio-{index}.m4a"),
                        remove_temp=True,
                        audio=True,
                        preset='medium',
                        ffmpeg_params=[
                            "-strict", "-2",
                            "-c:a", "aac",
                            "-b:a", "192k",
                            "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart"
                        ]
                    )

                    clips.append({
                        "file": output_filename,
                        "title": title,
                        "description": moment.get('description', ''),
                        "text": moment['text']
                    })
                    logging.info(f"Clip {index} created successfully")

                except Exception as e:
                    logging.error(f"Error creating clip {index}: {str(e)}", exc_info=True)

            return clips

    except Exception as e:
        logging.error(f"Error in create_clips: {str(e)}", exc_info=True)
        return []