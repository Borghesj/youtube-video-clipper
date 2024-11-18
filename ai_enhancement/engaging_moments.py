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
    # Fix the pattern construction
    pattern = '|'.join(f'({marker})' for marker in transition_markers)
    
    try:
        # Add flags parameter to re.finditer instead of inline
        for match in re.finditer(pattern, text, re.IGNORECASE):
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

    comedy_patterns = [
        # Setup patterns
        r'I was thinking about',
        r'you know what\'s funny',
        r'imagine if',
        r'picture this',
        r'you ever notice',
        # Punchline indicators
        r'like',
        r'([hH]a){2,}',  # Catches laughter
        r'holy shit',
        r'what the fuck',
        # Audience reactions
        r'\*laughter\*',
        r'\*applause\*',
        # Story markers
        r'so there I was',
        r'one time',
        r'the other day'
    ]
    
     # Increase scores for comedy markers
    for pattern in comedy_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            score += 2.0
            categories.add('comedy')
    
    # Add bonus for multiple patterns
    if score > 4:
        score *= 1.5

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

def analyze_narrative_coherence(text, prev_text="", next_text=""):
    """Analyze the narrative coherence of a text segment"""
    coherence_score = 0
    
    # Check for complete thoughts
    sentences = re.split(r'[.!?]+', text)
    complete_thought_ratio = sum(1 for s in sentences if len(s.split()) > 3) / max(len(sentences), 1)
    coherence_score += complete_thought_ratio * 2
    
    # Check for story elements
    story_elements = {
        'setup': ['first', 'so', 'there was', 'one time', 'let me tell you'],
        'context': ['because', 'since', 'at the time', 'back then'],
        'progression': ['then', 'after that', 'next', 'later'],
        'conclusion': ['finally', 'in the end', 'that\'s why', 'so yeah']
    }
    
    for element_type, markers in story_elements.items():
        if any(marker in text.lower() for marker in markers):
            coherence_score += 0.5
    
    # Check for topic consistency
    def get_key_terms(text):
        words = text.lower().split()
        return set(w for w in words if len(w) > 3)
    
    current_terms = get_key_terms(text)
    
    if prev_text:
        prev_terms = get_key_terms(prev_text)
        overlap = len(current_terms.intersection(prev_terms)) / max(len(current_terms), 1)
        coherence_score += overlap
    
    if next_text:
        next_terms = get_key_terms(next_text)
        overlap = len(current_terms.intersection(next_terms)) / max(len(current_terms), 1)
        coherence_score += overlap
    
    # Penalize for narrative breaks
    narrative_breaks = [
        'anyway',
        'moving on',
        'speaking of',
        'on another note',
        'by the way'
    ]
    
    breaks_count = sum(1 for break_term in narrative_breaks if break_term in text.lower())
    coherence_score -= breaks_count * 0.5
    
    return max(0, coherence_score)

def analyze_dialogue_content(text, prev_text="", next_text=""):
    """Enhanced analyze_dialogue_content with narrative coherence"""
    score, categories = analyze_content_structure(text)
    
    # Add narrative coherence analysis
    coherence_score = analyze_narrative_coherence(text, prev_text, next_text)
    score *= (1 + coherence_score * 0.5)  # Weight coherence but don't overwhelm other factors
    
    # Minimum content requirements
    min_words = 20
    max_words = 200
    word_count = len(text.split())
    
    if word_count < min_words or word_count > max_words:
        score *= 0.5
    
    # Check for incomplete stories
    if any(phrase in text.lower() for phrase in ["to be continued", "part 1", "stay tuned"]):
        score *= 0.3
    
    return score, categories

def analyze_dialogue_content(text, prev_text="", next_text=""):
    """Enhanced analyze_dialogue_content with narrative coherence"""
    engagement_indicators = {
        'emotional': [
            'love', 'hate', 'angry', 'happy', 'sad', 'excited', 'scared', 'afraid',
            'nervous', 'worried', 'proud', 'disgusted', 'surprised', 'confused',
            'wtf', 'omg', 'oh my god', 'holy', 'jeez', 'crazy', 'wow', 'amazing',
            'fuck', 'shit', 'damn', 'hell',
            'passionate', 'furious', 'thrilled', 'devastated', 'ecstatic', 'terrified'
        ],
        'conflict': [
            'fight', 'argue', 'disagree', 'wrong', 'no', 'never', 'stop', 'quit',
            'problem', 'issue', 'mistake', 'fault', 'blame', 'versus', 'against',
            'shut up', 'idiot', 'stupid',
            'war', 'battle', 'conflict', 'debate', 'challenge', 'face off'
        ],
        'humor': [
            'laugh', 'joke', 'funny', 'hilarious', 'ridiculous', 'crazy', 'silly',
            'weird', 'strange', 'bizarre', 'lmao', 'lol', 'haha', 'giggle',
            'dude', 'like', 'fuck', 'fucking', 'literally', 'actually', 'basically',
            'imagine if', 'picture this', 'so there I was',
            'absurd', 'insane', 'wild', 'nuts', 'crack up', 'lose it'
        ],
        'dramatic': [
            'sudden', 'suddenly', 'shocking', 'incredible', 'amazing', 'unbelievable',
            'never', 'ever', 'must', 'need', 'have to', 'critical', 'urgent',
            'emergency', 'crisis', 'dramatic', 'holy shit', 'what the fuck',
            'intense', 'extreme', 'life-changing', 'unforgettable'
        ],
        'informative': [
            'explain', 'because', 'reason', 'therefore', 'basically', 'essentially',
            'important', 'key', 'main', 'crucial', 'significant', 'actually', 'fact',
            'research', 'study', 'evidence', 'proves', 'demonstrates'
        ],
        'storytelling': [
            'my dad', 'my mom', 'my sister', 'my family', 'my friend',
            'I was', 'we were', 'there was', 'one time', 'back then',
            'at the time', 'that\'s when', 'so basically', 'long story short'
        ],
        'revelation': [
            'realized', 'discovered', 'figured out', 'turns out', 'plot twist',
            'mind blown', 'changed everything', 'never knew', 'found out'
        ],
        'controversy': [
            'controversial', 'debate', 'argument', 'politics', 'scandal',
            'conspiracy', 'outrage', 'protest', 'viral', 'trending'
        ],
        'personal': [
            'honestly', 'personally', 'in my opinion', 'for me', 'I think',
            'I believe', 'my experience', 'from my perspective'
        ],
        'action': [
            'running', 'jumping', 'fighting', 'racing', 'chasing',
            'explosion', 'crash', 'bang', 'boom', 'smash'
        ],
        'suspense': [
            'suddenly', 'unexpected', 'twist', 'surprise', 'out of nowhere',
            'wait for it', 'you won\'t believe', 'plot twist'
        ]
    }
    
    score = 0
    matched_categories = set()
    
    # Context analysis
    has_context = False
    if prev_text and any(word in text.lower() for word in ['so', 'then', 'because', 'but', 'and']):
        has_context = True
        score += 0.3

    if next_text and any(word in next_text.lower() for word in ['replied', 'answered', 'responded', 'said']):
        has_context = True
        score += 0.3

    # Add narrative coherence analysis
    coherence_score = analyze_narrative_coherence(text, prev_text, next_text)
    score += coherence_score * 0.5  # Weight coherence but don't overwhelm other factors

    # Enhanced engagement indicator check with category-specific multipliers
    for category, keywords in engagement_indicators.items():
        category_matches = sum(1 for word in keywords if word.lower() in text.lower())
        if category_matches > 0:
            multiplier = 0.2
            if category in ['humor', 'storytelling', 'revelation']:
                multiplier = 0.3
            elif category in ['dramatic', 'controversy']:
                multiplier = 0.3
            elif category in ['action', 'suspense']:
                multiplier = 0.3
            score += category_matches * 0.15 * multiplier
            matched_categories.add(category)

    # Combination bonus
    if len(matched_categories) >= 3:
        score *= 1.2  # Bonus for multi-category content
    
    # Viral patterns
    viral_patterns = ['wait for it', 'watch this', 'you won\'t believe', 'mind blown']
    if any(pattern in text.lower() for pattern in viral_patterns):
        score += 1.5

    # Maintain existing functionality
    structure_score, structure_categories = analyze_content_structure(text)
    score += structure_score
    matched_categories.update(structure_categories)
    
    boundaries = identify_narrative_boundaries(text)
    if boundaries:
        score += len(boundaries) * 0.5
    
    if '?' in text and next_text:
        score += 0.4
    if prev_text and '?' in prev_text:
        score += 0.3

    if '"' in text or '"' in text or '"' in text:
        score += 0.3
    
    exclamation_count = text.count('!')
    score += min(exclamation_count * 0.2, 0.4)
    
    filler_words = ['um', 'uh', 'like', 'you know', 'right', 'okay', 'basically', 'literally']
    if any(word in text.lower() for word in filler_words):
        score += 0.2

    story_markers = ['first', 'then', 'next', 'finally', 'after that', 'in the end']
    story_marker_count = sum(1 for marker in story_markers if marker in text.lower())
    if story_marker_count > 0:
        score += story_marker_count * 0.1
    
    # Length and coherence checks
    words = text.split()
    word_count = len(words)
    
    # Penalize very short or very long segments
    if word_count < 20:
        score *= 0.5
    elif word_count > 200:
        score *= 0.7
    elif 40 <= word_count <= 100:  # Ideal length range
        score *= 1.2
    
    if has_context:
        score *= 1.1

    # Penalize incomplete narratives
    if any(phrase in text.lower() for phrase in ["to be continued", "part 1", "stay tuned"]):
        score *= 0.3

    print(f"\nSegment: {text[:100]}...")
    print(f"Score: {score}")
    print(f"Categories: {matched_categories}")
    print(f"Word count: {word_count}")

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
        
        # Check for continuation phrases
        continuation_phrases = ['and then', 'but then', 'so then', 'because', 'however']
        has_continuation = any(phrase in next_moment['text'].lower() for phrase in continuation_phrases)
        
        # Check for incomplete quotes
        current_quotes = current['text'].count('"') % 2
        next_quotes = next_moment['text'].count('"') % 2
        has_incomplete_quote = current_quotes != 0 or next_quotes != 0
        
        # Check for comedy continuity
        has_comedy_continuation = (
            'comedy' in current.get('categories', set()) and 
            'comedy' in next_moment.get('categories', set())
        )
        
        time_gap = next_moment['start'] - current['end']
        merged_duration = next_moment['end'] - current['start']
        
        # Enhanced merging conditions
        should_merge = (
            time_gap <= max_gap and 
            merged_duration <= 120 and  # Increased from 90 to 120 for comedy segments
            (not is_strong_break or has_continuation or has_incomplete_quote or has_comedy_continuation)
        )
        
        if should_merge:
            # Merge with dynamic padding
            padding = 0.5 if has_continuation else 0.3
            if has_comedy_continuation:
                padding += 0.2  # Extra padding for comedy timing
            
            current['end'] = max(current['end'], next_moment['end']) + padding
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
            # Only add if it meets minimum quality criteria
            if current['end'] - current['start'] >= 3 and not has_incomplete_quote:
                merged.append(current)
            current = next_moment.copy()
            # Add lead-in padding for new segments
            current['start'] = max(0, current['start'] - 0.3)
    
    # Handle final moment
    if current['end'] - current['start'] >= 3:
        # Add extra padding to final moment if it has a conclusion
        if any(x in current['text'].lower() for x in ['finally', 'in conclusion', 'that\'s why']):
            current['end'] += 0.3
        merged.append(current)
    
    return merged

def identify_engaging_moments_internal(video_path, transcript, video_name):
    """Identify engaging moments with improved context awareness"""
    logging.info(f"Identifying engaging moments for video: {video_name}")
    print(f"\nProcessing video: {video_name}")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps
    
    # Initial split by sentence boundaries
    raw_segments = re.split(r'(?<=[.!?])\s+(?=[A-Z])', transcript)
    
    # Further split long segments
    refined_segments = []
    for segment in raw_segments:
        if len(segment.split()) > 100:  # If segment is too long
            # Split by simpler patterns
            subsegments = []
            temp = segment
            
            # Split by comma-space
            parts = [p.strip() for p in temp.split(', ')]
            
            # Further split long parts by common conjunctions
            for part in parts:
                if len(part.split()) > 50:
                    # Split by common joining words
                    subparts = []
                    for conjunction in [' and ', ' but ', ' so ', ' then ']:
                        if conjunction in part:
                            subparts.extend([p.strip() for p in part.split(conjunction)])
                            break
                    if not subparts:  # If no splits happened
                        subparts = [part]
                    subsegments.extend(subparts)
                else:
                    subsegments.append(part)
            
            # Add valid subsegments
            valid_subsegments = [s for s in subsegments if s and len(s.split()) > 10]
            refined_segments.extend(valid_subsegments)
        else:
            refined_segments.append(segment)
    
    raw_segments = [s for s in refined_segments if s.strip()]  # Update raw_segments with refined ones
    print(f"Found {len(raw_segments)} segments to analyze")
    segments = []
    
    # Process segments with context
    for i, segment in enumerate(raw_segments):
        prev_segment = raw_segments[i-1] if i > 0 else ""
        next_segment = raw_segments[i+1] if i < len(raw_segments)-1 else ""
        
        score, categories = analyze_dialogue_content(segment, prev_segment, next_segment)
        print(f"\nSegment {i}:")
        print(f"Text: {segment[:100]}...")
        print(f"Score: {score}")
        print(f"Categories: {categories}")
        
        if score > 1.0:  # Using your threshold
            print("Segment accepted!")
            segments.append({
                'text': segment,
                'score': score,
                'categories': categories,
                'index': i
            })
        else:
            print("Segment rejected - score too low")
    
    print(f"\nAccepted {len(segments)} segments")
    
    # Keep all existing moment processing logic
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
            if segment['score'] >= 2.0:
                current_moment = {
                    'segments': [segment],
                    'score': segment['score'],
                    'categories': segment['categories'],
                    'start_index': segment['index']
                }
                print(f"\nStarting new moment with score {segment['score']}")
        else:
            # Calculate potential duration
            potential_text = ' '.join(s['text'] for s in current_moment['segments']) + ' ' + segment['text']
            start_text = ' '.join(raw_segments[:current_moment['start_index']])
            words_before = len(start_text.split())
            words_per_second = len(transcript.split()) / total_duration
            potential_duration = len(potential_text.split()) / words_per_second
            
            context_score = len(current_moment['categories'].intersection(segment['categories']))
            
            # Check if adding this segment would make the moment too long
            if potential_duration <= 90 and (context_score > 0 or segment['score'] >= 1.8):
                current_moment['segments'].append(segment)
                current_moment['score'] += segment['score']
                current_moment['categories'].update(segment['categories'])
                print(f"Extended moment - new score: {current_moment['score']}")
            else:
                # Process current moment if it's valid
                processed_moment = process_current_moment(
                    current_moment,
                    len(' '.join(raw_segments[:current_moment['start_index']]).split())
                )
                if processed_moment:
                    engaging_moments.append(processed_moment)
                    print(f"Added moment with score {current_moment['score']}")
                
                # Start new moment if segment is engaging enough
                if segment['score'] >= 1.8:
                    current_moment = {
                        'segments': [segment],
                        'score': segment['score'],
                        'categories': segment['categories'],
                        'start_index': segment['index']
                    }
                    print(f"Starting new moment with score {segment['score']}")
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
            print("Added final moment")
    
    cap.release()
    
    print(f"\nFound {len(engaging_moments)} engaging moments before merging")
    
    if engaging_moments:
        # Merge overlapping moments with adjusted parameters
        merged_moments = merge_overlapping_moments(engaging_moments, max_gap=2.0)
        logging.info(f"Found {len(merged_moments)} engaging moments")
        print(f"Final count after merging: {len(merged_moments)} moments")
        
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

def generate_tiktok_description(video_name: str, transcript: str, moment_text: str) -> dict:
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
        import json
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
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_transcript = "Your test transcript here"
    moments = identify_engaging_moments("video_path", test_transcript, "Test Video")
    for moment in moments:
        print(f"Found moment: {moment['title']}")