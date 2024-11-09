import logging
import os
import cv2
import numpy as np
import tensorflow as tf
import openai
from config import OPENAI_API_KEY, FACE_CASCADE_PATH, SENTIMENT_MODEL_PATH
import time
import threading
import concurrent.futures
import re

# Set up OpenAI API key
openai.api_key = OPENAI_API_KEY

# Load pre-trained models
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH) if FACE_CASCADE_PATH else None

# Try to load sentiment model, but continue if it's not found
sentiment_model = None
if SENTIMENT_MODEL_PATH and os.path.exists(SENTIMENT_MODEL_PATH):
    try:
        sentiment_model = tf.keras.models.load_model(SENTIMENT_MODEL_PATH)
    except Exception as e:
        logging.warning(f"Failed to load sentiment model: {e}")
else:
    logging.warning(f"Sentiment model not found at {SENTIMENT_MODEL_PATH}")

def analyze_frame(frame):
    """Analyze a single frame for faces and sentiment"""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4) if face_cascade else []
        sentiment_score = 0
        if sentiment_model:
            resized_frame = cv2.resize(frame, (224, 224))
            normalized_frame = resized_frame / 255.0
            sentiment_score = sentiment_model.predict(np.expand_dims(normalized_frame, axis=0))[0][0]
        return len(faces), float(sentiment_score)
    except Exception as e:
        logging.error(f"Error in analyze_frame: {str(e)}", exc_info=True)
        return 0, 0.0

def identify_narrative_boundaries(text):
    """Identify strong narrative transitions in content"""
    transition_markers = [
        r'anyway[,.]',
        r'moving on',
        r'speaking of',
        r'but (?:um|uh|you know)',
        r'so[,.]',
        r'now[,.]',
        r'here\'s the thing',
        r'let me tell you',
        r'talk about',
        r'let\'s talk about',
        r'i\'ll tell you',
        r'you know what\'s funny',
        r'on another note',
        r'switching gears',
        r'changing topics'
    ]
    
    boundaries = []
    pattern = '|'.join(f'(?i){marker}' for marker in transition_markers)
    
    try:
        for match in re.finditer(pattern, text):
            boundaries.append(match.start())
    except Exception as e:
        logging.error(f"Error in regex pattern matching: {str(e)}")
        return []
    
    return sorted(boundaries)

def analyze_content_structure(text):
    """Analyze content structure for various patterns"""
    score = 0
    categories = set()
    
    setup_patterns = [
        r'you (?:know|remember|ever)',
        r'when (?:I|you|we)',
        r'there\'s this|there was this',
        r'so (?:I\'m|we\'re)',
        r'imagine if|picture this'
    ]
    
    punchline_patterns = [
        r'but then|and then',
        r'[!?]{2,}',
        r'that\'s when|that\'s how',
        r'turns out',
        r'plot twist'
    ]
    
    reaction_patterns = [
        r'laugh',
        r'clap',
        r'applause',
        r'response',
        r'reaction'
    ]
    
    informative_patterns = [
        r'because',
        r'therefore',
        r'this means',
        r'the reason is',
        r'essentially',
        r'basically'
    ]
    
    # Add case-insensitive pattern matching
    for pattern in setup_patterns:
        if re.search(f'(?i){pattern}', text):
            score += 1.5
            categories.add('setup')
    
    for pattern in punchline_patterns:
        if re.search(f'(?i){pattern}', text):
            score += 2
            categories.add('punchline')
    
    for pattern in reaction_patterns:
        if re.search(f'(?i){pattern}', text):
            score += 1
            categories.add('audience')
    
    for pattern in informative_patterns:
        if re.search(f'(?i){pattern}', text):
            score += 1.5
            categories.add('informative')
    
    if '?' in text and '!' in text:
        score += 1
    
    return score, categories

def analyze_dialogue_content(text, prev_text="", next_text=""):
    """Analyze dialogue content with comprehensive detection"""
    engagement_indicators = {
        'emotional': [
            'love', 'hate', 'angry', 'happy', 'sad', 'excited', 'scared', 'afraid',
            'nervous', 'worried', 'proud', 'disgusted', 'surprised', 'confused',
            'wtf', 'omg', 'oh my god', 'holy', 'jeez', 'crazy', 'wow', 'amazing'
        ],
        'conflict': [
            'fight', 'argue', 'disagree', 'wrong', 'no', 'never', 'stop', 'quit',
            'problem', 'issue', 'mistake', 'fault', 'blame', 'versus', 'against',
            'shut up', 'idiot', 'stupid'
        ],
        'humor': [
            'laugh', 'joke', 'funny', 'hilarious', 'ridiculous', 'crazy', 'silly',
            'weird', 'strange', 'bizarre', 'lmao', 'lol', 'haha', 'giggle'
        ],
        'dramatic': [
            'sudden', 'suddenly', 'shocking', 'incredible', 'amazing', 'unbelievable',
            'never', 'ever', 'must', 'need', 'have to', 'critical', 'urgent',
            'emergency', 'crisis', 'dramatic'
        ],
        'informative': [
            'explain', 'because', 'reason', 'therefore', 'basically', 'essentially',
            'important', 'key', 'main', 'crucial', 'significant', 'actually', 'fact'
        ]
    }
    
    score = 0
    matched_categories = set()
    
    # Context analysis
    has_context = False
    if prev_text and any(word in text.lower() for word in ['so', 'then', 'because', 'but', 'and']):
        has_context = True
        score += 1
    
    if next_text and any(word in next_text.lower() for word in ['replied', 'answered', 'responded', 'said']):
        has_context = True
        score += 1

    # Check engagement indicators
    for category, keywords in engagement_indicators.items():
        category_matches = sum(1 for word in keywords if word.lower() in text.lower())
        if category_matches > 0:
            score += category_matches * 0.75
            matched_categories.add(category)
    
    # Add content structure analysis
    structure_score, structure_categories = analyze_content_structure(text)
    score += structure_score
    matched_categories.update(structure_categories)
    
    # Check for narrative transitions
    boundaries = identify_narrative_boundaries(text)
    if boundaries:
        score += len(boundaries) * 0.5
    
    # Question-answer patterns
    if '?' in text and next_text:
        score += 1.5
    if prev_text and '?' in prev_text:
        score += 1

    # Quotations and dialogue
    if '"' in text or '"' in text or '"' in text:
        score += 1
    
    # Exclamations and emphasis
    exclamation_count = text.count('!')
    score += min(exclamation_count * 0.5, 1.5)
    
    # Natural speech patterns
    filler_words = ['um', 'uh', 'like', 'you know', 'right']
    if any(word in text.lower() for word in filler_words):
        score += 0.5
    
    if has_context:
        score *= 1.2
    
    return score, matched_categories

def merge_overlapping_moments(moments, max_gap=2.0):
    """Merge moments while preserving context and structure"""
    if not moments:
        return []
    
    valid_moments = [m for m in moments if m.get('text', '').strip()]
    if not valid_moments:
        return []
    
    merged = []
    current = valid_moments[0].copy()
    
    for next_moment in valid_moments[1:]:
        # Check for strong narrative breaks
        boundaries = identify_narrative_boundaries(next_moment['text'])
        is_strong_break = bool(boundaries)
        
        time_gap = next_moment['start'] - current['end']
        merged_duration = next_moment['end'] - current['start']
        
        # Merging conditions
        if (time_gap <= max_gap and 
            merged_duration <= 90 and  # 1.5 minutes max
            not is_strong_break):
            
            # Merge with padding
            current['end'] = max(current['end'], next_moment['end']) + 0.3
            current['text'] = current['text'] + ' ' + next_moment['text']
            current['score'] = max(current.get('score', 0), next_moment.get('score', 0))
            current['faces'] = max(current.get('faces', 0), next_moment.get('faces', 0))
            current['sentiment'] = (current.get('sentiment', 0) + next_moment.get('sentiment', 0)) / 2
            
            if 'categories' in current and 'categories' in next_moment:
                if isinstance(current['categories'], list):
                    current['categories'] = list(set(current['categories'] + next_moment['categories']))
                else:
                    current['categories'] = list(current['categories'].union(next_moment['categories']))
        else:
            if current['end'] - current['start'] >= 3:  # Minimum duration check
                merged.append(current)
            current = next_moment.copy()
            current['start'] = max(0, current['start'] - 0.2)
    
    if current['end'] - current['start'] >= 3:
        merged.append(current)
    
    return merged

def identify_engaging_moments_internal(video_path, transcript, video_name):
    """Identify engaging moments with improved context awareness"""
    logging.info(f"Identifying engaging moments for video: {video_name}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps
    
    # Split into segments
    raw_segments = re.split(r'(?<=[.!?])\s+(?=[A-Z])', transcript)
    segments = []
    
    # Process segments with context
    for i, segment in enumerate(raw_segments):
        prev_segment = raw_segments[i-1] if i > 0 else ""
        next_segment = raw_segments[i+1] if i < len(raw_segments)-1 else ""
        
        score, categories = analyze_dialogue_content(segment, prev_segment, next_segment)
        
        if score > 1.8:  # Threshold for initial segment inclusion
            segments.append({
                'text': segment,
                'score': score,
                'categories': categories,
                'index': i
            })
    
    engaging_moments = []
    current_moment = None
    
    def process_current_moment(moment, words_before):
        """Helper function to process and create a moment"""
        if not moment or len(moment['segments']) < 2:
            return None
            
        moment_text = ' '.join(s['text'] for s in moment['segments'])
        words_per_second = len(transcript.split()) / total_duration
        
        # Calculate timing with padding
        start_time = max(0, words_before / words_per_second - 0.3)
        end_time = (words_before + len(moment_text.split())) / words_per_second + 0.5
        
        return {
            'start': start_time,
            'end': end_time,
            'text': moment_text,
            'score': moment['score'],
            'categories': list(moment['categories']),
            'faces': 0,
            'sentiment': 0.0
        }
    
    for segment in segments:
        if current_moment is None:
            if segment['score'] >= 1.8:
                current_moment = {
                    'segments': [segment],
                    'score': segment['score'],
                    'categories': segment['categories'],
                    'start_index': segment['index']
                }
        else:
            # Calculate potential duration
            potential_text = ' '.join(s['text'] for s in current_moment['segments']) + ' ' + segment['text']
            start_text = ' '.join(raw_segments[:current_moment['start_index']])
            words_before = len(start_text.split())
            words_per_second = len(transcript.split()) / total_duration
            potential_duration = len(potential_text.split()) / words_per_second
            
            context_score = len(current_moment['categories'].intersection(segment['categories']))
            
            # Check if adding this segment would make the moment too long
            if potential_duration <= 90 and (context_score > 0 or segment['score'] >= 1.8):  # 1.5 minutes max
                current_moment['segments'].append(segment)
                current_moment['score'] += segment['score']
                current_moment['categories'].update(segment['categories'])
            else:
                # Process current moment if it's valid
                processed_moment = process_current_moment(
                    current_moment,
                    len(' '.join(raw_segments[:current_moment['start_index']]).split())
                )
                if processed_moment:
                    engaging_moments.append(processed_moment)
                
                # Start new moment if segment is engaging enough
                if segment['score'] >= 1.8:
                    current_moment = {
                        'segments': [segment],
                        'score': segment['score'],
                        'categories': segment['categories'],
                        'start_index': segment['index']
                    }
                else:
                    current_moment = None
    
    # Process final moment
    if current_moment:
        processed_moment = process_current_moment(
            current_moment,
            len(' '.join(raw_segments[:current_moment['start_index']]).split())
        )
        if processed_moment:
            engaging_moments.append(processed_moment)
    
    cap.release()
    
    if engaging_moments:
        # Merge overlapping moments with adjusted parameters
        merged_moments = merge_overlapping_moments(engaging_moments, max_gap=2.0)
        logging.info(f"Found {len(merged_moments)} engaging moments")
        
        try:
            prompt = f"""Analyze the following video transcript titled "{video_name}" and the identified engaging moments. For each moment, provide three lines in this exact format:
Title: [A catchy title that captures the essence of the moment]
Description: [A brief description of why it's engaging]
Text: [The exact text being spoken during this moment]

Key points to consider:
- Look for complete narrative arcs and dialogue exchanges
- Identify the setup and payoff in comedic moments
- Ensure the context makes sense on its own

Transcript segments to analyze:
{merged_moments}"""

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes video content and identifies engaging moments. Focus on complete narratives, dramatic moments, and maintaining context."},
                    {"role": "user", "content": prompt}
                ]
            )

            ai_descriptions = response.choices[0].message['content'].split('\n\n')
            final_moments = []
            
            for i, moment in enumerate(merged_moments):
                processed_moment = moment.copy()
                if i < len(ai_descriptions):
                    description = ai_descriptions[i]
                    lines = description.strip().split('\n')
                    for line in lines:
                        if line.startswith('Title: '):
                            processed_moment['title'] = line[7:].strip()
                        elif line.startswith('Description: '):
                            processed_moment['description'] = line[13:].strip()
                final_moments.append(processed_moment)
            
            return final_moments
        except Exception as e:
            logging.error(f"Error in OpenAI API call: {str(e)}")
            return merged_moments
    
    return []

def identify_engaging_moments_with_timeout(video_path, transcript, video_name, timeout=600):
    """Run identify_engaging_moments_internal with a timeout"""
    result = []
    exception = None

    def target():
        nonlocal result, exception
        try:
            result.extend(identify_engaging_moments_internal(video_path, transcript, video_name))
        except Exception as e:
            exception = e

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        logging.error(f"identify_engaging_moments function timed out after {timeout} seconds")
        return []
    if exception:
        raise exception
    return result

def identify_engaging_moments(video_path, transcript, video_name, timeout=600):
    """Main entry point for identifying engaging moments"""
    try:
        return identify_engaging_moments_with_timeout(video_path, transcript, video_name, timeout)
    except Exception as e:
        logging.error(f"Error in AI engaging moment identification: {str(e)}", exc_info=True)
        return []

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_transcript = "Your test transcript here"
    moments = identify_engaging_moments("video_path", test_transcript, "Test Video")
    for moment in moments:
        print(f"Found moment: {moment['title']}")