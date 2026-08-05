# -*- coding: utf-8 -*-
"""
IMAGE_EXTENDER_v0_1_1 - 01022026 Local Version

This is a local Python adaptation of the original Google Colab notebook.
It provides the same functionality but runs locally with a tkinter GUI.
"""

import os
import sys
import base64
import json
import re
import ast
import time
import threading
import random
import datetime
import csv
import numpy as np
import requests
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import io
import tempfile
import warnings
warnings.filterwarnings('ignore')

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# For audio processing
try:
    from pydub import AudioSegment
    from pydub.playback import play as pydub_play
    import soundfile as sf
    import librosa
    from scipy.signal import hilbert, butter, sosfiltfilt, sosfilt, iirpeak, lfilter, windows
    import scipy.signal as signal
    import pyloudnorm
    import pedalboard
    from pedalboard import Pedalboard, Reverb
    from pedalboard.io import AudioFile
    import pyroomacoustics as pra
    from nara_wpe.wpe import wpe
    from nara_wpe.utils import stft, istft
    import soundfile as sf
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError as e:
    print(f"Audio libraries not available: {e}")
    print("Some audio features may not work. Install with: pip install pydub scipy soundfile librosa pyloudnorm pedalboard")
    AUDIO_AVAILABLE = False

# For image processing
try:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    print("MediaPipe not available. Install with: pip install mediapipe opencv-python")
    MEDIAPIPE_AVAILABLE = False

# For OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    print("OpenAI not available. Install with: pip install openai")
    OPENAI_AVAILABLE = False

# ----------------------------------------------------------------------
# GLOBAL VARIABLES AND SETTINGS
# ----------------------------------------------------------------------

# API Keys (user should set these)
openai_api_key = ""
freesound_api_key = ""

# Image variables
IMAGE_FILE = None
uploaded_image = None
processed_image = None

# Object detection results
recognized_tags = []
sound_pannings = []
scene_and_location_tags = []
image_input_description = []
importance_values = []
room_detected = None
room_size = None
damping = None
wet_level = None
width = None

# Sound settings
saved_sound_settings = []
quality_mode = True
prefer_rating = True

# Audio processing
used_sound_ids = set()
downloaded_files = []
processed_tracks = []

# Background music settings
background_music_enabled = True
background_music_playing = False
background_music_thread = None
background_music_volume = 0.1  # 10% volume for background music
background_music_target_volume = 0  # Target volume for fading
background_music_current_volume = 0.1  # Current actual volume
background_music_file = "background_music.mp3"  # Name of your background music file
BACKGROUND_MUSIC_FADE_DURATION = 0.2  # 2 second fade out
current_active_tab = None  # Track which tab is currently active

# Reverb settings
reverb_enabled = False
current_mix_raw = None
current_mix_with_reverb = None
reverb_room_size = None
reverb_damping = None
reverb_wet_level = None
reverb_width = None

# Audio constants
BARK_BANDS = [20, 100, 200, 300, 400, 510, 630, 770, 920,
              1080, 1270, 1480, 1720, 2000, 2320, 2700,
              3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500, 20000]
MAX_GAIN_REDUCTION = 6
SMOOTHING_TIME = 0.1
TARGET_DBFS = -20.0
FADE_DURATION = 500
DESCRIPTION_CHECK_ENABLED = True
MAX_DESCRIPTION_CHECK_ROUNDS = 1
MAX_OPENAI_FALLBACK_TRIES = 3
ATMO_LOUDNESS_DBFS = -30.0
MIN_LOUDNESS_DBFS = -30.0
MAX_LOUDNESS_DBFS = -20.0

# GUI variables
current_rating = 0
is_playing = False
audio_thread = None

# Email settings
EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': '',  # Change this to your email
    'sender_password': '',  # Change this to your app password
    'receiver_email': 'image.extender.feedback@gmail.com' 
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

# ----------------------------------------------------------------------
# UTILITY FUNCTIONS
# ----------------------------------------------------------------------

# Setup directories function
def setup_directories():
    """Create necessary directories for the application"""
    import os
    
    # Get base directory
    if '__file__' in globals():
        base_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        base_dir = os.getcwd()
    
    # Create required directories
    directories = [
        "downloaded_sounds",
        "logs", 
        "temp_playback",
        "exports",
        "temp",
        "images"  
    ]
    
    print(f"\n{'='*50}")
    print(f"BASE DIRECTORY: {base_dir}")
    print(f"{'='*50}")
    
    for dir_name in directories:
        dir_path = os.path.join(base_dir, dir_name)
        try:
            os.makedirs(dir_path, exist_ok=True)
            print(f"✓ Directory created/verified: {dir_path}")
        except Exception as e:
            print(f"✗ Could not create directory {dir_name}: {e}")
    
    print(f"{'='*50}\n")
    
    return base_dir

# Run directory setup
BASE_DIR = setup_directories()

def play_background_music():
    """Play background music in a loop with sounddevice"""
    global background_music_playing, background_music_enabled
    global background_music_target_volume, background_music_current_volume, current_active_tab
    
    if not AUDIO_AVAILABLE:
        return
        
    # Check if music file exists
    music_path = os.path.join(BASE_DIR, background_music_file)
    if not os.path.exists(music_path):
        print(f"Background music file not found: {music_path}")
        return
    
    try:
        # Load audio file with soundfile
        data, samplerate = sf.read(music_path)
        
        # Convert to mono if stereo for easier mixing
        if len(data.shape) > 1 and data.shape[1] > 1:
            data = data.mean(axis=1)
        
        # Normalize audio
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val  # Normalize to 1.0
        
        background_music_playing = True
        
        # FIXED: Use sounddevice's callback mode for smooth continuous playback
        # Create a buffer that loops the audio
        audio_buffer = data.copy()
        current_position = 0
        
        def audio_callback(outdata, frames, time_info, status):
            nonlocal current_position, audio_buffer
            
            if status:
                print(f"Audio status: {status}")
            
            # Check if we need to restart the background music
            if not background_music_playing:
                outdata[:] = 0
                return
            
            # Calculate how much we need to copy
            frames_needed = frames
            outdata_filled = 0
            
            while frames_needed > 0:
                # Calculate samples available from current position to end
                samples_available = len(audio_buffer) - current_position
                
                if samples_available <= 0:
                    # Reached end, loop back to beginning
                    current_position = 0
                    samples_available = len(audio_buffer) - current_position
                
                # Copy what we can
                samples_to_copy = min(samples_available, frames_needed)
                
                # Apply current volume
                volume_adjusted = audio_buffer[current_position:current_position + samples_to_copy] * background_music_current_volume
                
                # Handle stereo output (copy mono to both channels)
                if outdata.shape[1] == 2:  # Stereo output
                    outdata[outdata_filled:outdata_filled + samples_to_copy, 0] = volume_adjusted
                    outdata[outdata_filled:outdata_filled + samples_to_copy, 1] = volume_adjusted
                else:  # Mono output
                    outdata[outdata_filled:outdata_filled + samples_to_copy, 0] = volume_adjusted
                
                # Update positions
                current_position += samples_to_copy
                frames_needed -= samples_to_copy
                outdata_filled += samples_to_copy
        
        # Fade in if starting from silent
        if background_music_current_volume < 0.001:
            print("Starting background music with fade in...")
            fade_steps = 50
            for i in range(fade_steps + 1):
                if not background_music_playing:
                    break
                background_music_current_volume = (background_music_target_volume * i) / fade_steps
                time.sleep(0.02)  # ~1 second fade in
        
        print(f"Starting background music stream at samplerate {samplerate}")
        
        # Create and start the stream with callback
        stream = sd.OutputStream(
            samplerate=samplerate,
            channels=2,  # Always output stereo
            callback=audio_callback,
            blocksize=1024,  # Reasonable block size
            finished_callback=lambda: print("Background music stream finished")
        )
        
        stream.start()
        
        # Main loop to handle volume changes
        while background_music_playing:
            try:
                # Smoothly adjust to target volume
                if abs(background_music_current_volume - background_music_target_volume) > 0.001:
                    # Move 10% closer to target each iteration
                    background_music_current_volume += (background_music_target_volume - background_music_current_volume) * 0.1
                
                # Check if we should stop (target volume is 0 and we've faded out)
                if background_music_target_volume < 0.001 and background_music_current_volume < 0.001:
                    print("Background music faded out completely")
                    break
                    
                time.sleep(0.1)  # Check volume changes every 100ms
                
            except Exception as e:
                print(f"Error in background music loop: {e}")
                time.sleep(0.1)
        
        # Cleanup
        print("Stopping background music stream...")
        if stream.active:
            stream.stop()
        stream.close()
        
        print("Background music stopped")
        
    except Exception as e:
        print(f"Background music error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        background_music_playing = False
        background_music_current_volume = 0.0

def stop_background_music(immediate=False):
    """Stop background music gracefully with fade out or immediately"""
    global background_music_playing, background_music_target_volume, background_music_current_volume
    
    if not background_music_playing:
        return
    
    if immediate:
        # Immediate stop (for app shutdown)
        print("Stopping background music immediately...")
        background_music_playing = False
        background_music_target_volume = 0.0
        background_music_current_volume = 0.0
        try:
            # Stop all sounddevice playback
            sd.stop()
        except:
            pass
        print("Background music stopped immediately")
    else:
        # Gradual fade out
        print(f"Starting {BACKGROUND_MUSIC_FADE_DURATION}s fade out...")
        
        # Set target volume to 0 for fade out
        background_music_target_volume = 0.0
        
        # Wait for the fade to complete
        fade_steps = int(BACKGROUND_MUSIC_FADE_DURATION * 10)  # Check 10 times per second
        for i in range(fade_steps):
            if not background_music_playing:
                break
            
            # Check if volume has reached near zero
            if background_music_current_volume < 0.001:
                break
                
            time.sleep(0.1)  # Check every 100ms
        
        # Now actually stop
        background_music_playing = False
        
        # Stop playback
        try:
            sd.stop()
        except:
            pass
        
        # Reset volume for next time
        background_music_current_volume = 0.0
        
        print(f"Background music stopped after fade")
    

def start_background_music():
    """Start or resume background music"""
    global background_music_playing, background_music_thread, background_music_enabled
    global background_music_target_volume, background_music_current_volume, current_active_tab
    
    # Don't start if we're in Sound Creation tab
    if current_active_tab == "Sound Creation":
        print("Not starting background music - currently in Sound Creation tab")
        return
    
    if background_music_playing:
        # Music is already playing, just adjust volume if needed
        if background_music_target_volume < background_music_volume:
            print("Resuming background music volume")
            background_music_target_volume = background_music_volume
        return
        
    # Stop any existing thread first
    if background_music_thread and background_music_thread.is_alive():
        try:
            background_music_playing = False
            background_music_thread.join(timeout=1.0)
        except:
            pass
    
    # Reset volumes
    background_music_current_volume = 0.0
    background_music_target_volume = background_music_volume
    background_music_enabled = True
    
    # Start new thread
    background_music_thread = threading.Thread(
        target=play_background_music,
        daemon=True,
        name="BackgroundMusicThread"
    )
    background_music_thread.start()
    print("Background music started")

def toggle_background_music_for_tab(tab_name):
    """Control background music based on active tab"""
    global background_music_enabled, background_music_target_volume, current_active_tab
    
    # If switching to the same tab, do nothing
    if tab_name == current_active_tab:
        return
    
    previous_tab = current_active_tab
    current_active_tab = tab_name
    
    print(f"Tab change: {previous_tab} -> {tab_name}")
    
    #TODO MAYBE ERROR POTENZIAL
    '''
    # Case 1: Switching TO Sound Creation tab (check if processing)
    if tab_name == "Sound Creation":
        # Check if we're currently processing sounds (download button is disabled)
        if hasattr(app, 'download_button') and app.download_button.cget('state') == tk.DISABLED:
            print("Sound Creation tab - but processing is active, keeping background music")
            # Don't fade out - keep music playing during processing
            background_music_target_volume = background_music_volume
        else:
            print("Sound Creation tab - fading out background music")
            # Fade out only if not processing
            background_music_target_volume = 0.0
    '''
    # Case 2: Switching FROM Sound Creation tab to any other tab (fade in)
    if previous_tab == "Sound Creation":
        print(f"Leaving Sound Creation tab - fading in background music for {tab_name}")
        background_music_target_volume = background_music_volume
    
    # Case 3: Switching between non-Sound-Creation tabs (keep music playing, no changes)
    else:
        # Just update tab tracking, music continues uninterrupted
        print(f"Switching between non-sound tabs: {previous_tab} -> {tab_name}")
        # Ensure music is playing if it should be
        if not background_music_playing and background_music_enabled:
            start_background_music()

def encode_image_to_base64(image_path):
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def parse_gpt_output(output_text):
    """Parse GPT output safely to Python dict"""
    cleaned = re.sub(r"^```(?:python)?\s*|\s*```$", "", output_text.strip(), flags=re.IGNORECASE)
    try:
        return ast.literal_eval(cleaned)
    except Exception:
        try:
            return json.loads(cleaned)
        except Exception:
            return {"objekte": [], "szenerie_und_ort": [], "stimmung_und_komposition": [], "panning": []}

def load_tag_hierarchy(csv_path=None):
    """Load tag hierarchy from CSV file"""
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), "tag_hierarchy.csv")
    
    hierarchy = []
    try:
        if os.path.exists(csv_path):
            with open(csv_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if row:
                        hierarchy.append([col.strip().lower() for col in row])
    except Exception as e:
        print(f"Warning: Could not load tag hierarchy: {e}")
    return hierarchy

def get_fallback_tag(tag, hierarchy_data):
    """Get fallback tag from hierarchy"""
    tag = tag.lower()
    for row in hierarchy_data:
        if tag in row:
            index = row.index(tag)
            if index > 0:
                return row[index - 1]
    return None

def check_description_with_openai(description, tag, image_description):
    """Check if sound description matches tag using OpenAI"""
    if not OPENAI_AVAILABLE:
        print("OpenAI not available, skipping description check")
        return True
        
    try:
        client = openai.OpenAI(api_key=openai_api_key)
        
        prompt = f"""
        Analyze if the sound description accurately represents the object/tag "{tag}" in the context of the image described as: "{image_description}".

        Sound Description: "{description}"
        Tag to check: "{tag}"
        Image Description: "{image_description}"

        Return ONLY "MATCH" if the sound clearly represents the specified object/tag,
        or "NO_MATCH" if the description mentions the word but the sound doesn't actually represent that object. In case of art/paintings you can be more open minded and also accept atmospheric/abstract sounds that could match the vibe.

        Consider:
        - Does the sound directly represent the object? (e.g., "car engine" for "car")
        - Or is the word just mentioned in context? (e.g., "song about cars" for "car")
        - Is it the actual sound of the object or just a reference to it?

        Be strict - only return MATCH for clear, direct representations.
        """

        print(f"\n=== Description Check ===")
        print(f"Tag: '{tag}'")
        print(f"Description: '{description}'")
        print(f"Image Context: '{image_description}'")

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt}
                ]
            }],
            max_output_tokens=16,
        )

        result = response.output_text.strip().upper()
        print(f"OpenAI response: {result}")

        return result == "MATCH"
        
    except Exception as e:
        print(f"Error in OpenAI description check: {e}")
        return True

def get_openai_fallback_tag(tag):
    """Get fallback tag from OpenAI"""
    if not OPENAI_AVAILABLE:
        return None
        
    try:
        client = openai.OpenAI(api_key=openai_api_key)
        
        prompt = f"""
        Find a suitable sound-related synonym or broader ontological category for the tag "{tag}" that would be
        good for finding similar sounds on freesound.org.

        Return ONLY the single best alternative tag as a plain text string, nothing else.
        The alternative should be:
        - Semantically related to "{tag}"
        - Broader or more general if possible
        - Useful for sound search
        - A single word or short phrase

        Examples:
        - "bicycle" -> "vehicle"
        - "piano" -> "keyboard"
        - "angry" -> "emotion"
        - "ash" -> "tree"
        """

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt}
                ]
            }],
            max_output_tokens=50,
        )

        fallback_tag = response.output_text.strip()
        fallback_tag = re.sub(r'^["\']|["\']$', '', fallback_tag)
        fallback_tag = fallback_tag.split('\n')[0].strip()

        if fallback_tag and fallback_tag.lower() != tag.lower():
            print(f"OpenAI Fallback for '{tag}': '{fallback_tag}'")
            return fallback_tag
        else:
            print(f"No suitable OpenAI fallback for '{tag}'")
            return None
            
    except Exception as e:
        print(f"Error in OpenAI fallback: {e}")
        return None
    
def design_eq_for_masking(masker_energy, maskee_energy, filterbank):
    """EQ settings for masking reduction"""
    eq_settings = []
    print("\nMasking Analysis Results:")
    print(f"{'Freq':<8} {'Masker(dB)':<10} {'Target(dB)':<10} {'Diff(dB)':<10} {'Action':<12}")
    print("-" * 50)

    valid_bands = 0
    adjustments_made = False

    for (f1, e1), (f2, e2), (_, cf) in zip(masker_energy, maskee_energy, filterbank):
        # Skip invalid bands
        if np.isnan(e1) or np.isnan(e2) or np.isinf(e1) or np.isinf(e2):
            continue

        valid_bands += 1
        diff_db = e1 - e2

        # Only consider bands where both signals are present
        if e1 > -80 and e2 > -80:
            # Only apply EQ if masker is significantly louder
            if diff_db > 8:  # Threshold for masking detection
                reduction = min(MAX_GAIN_REDUCTION, 0.3 * diff_db)
                eq_settings.append({
                    'type': 'peak',
                    'freq': cf,
                    'gain': -reduction,
                    'q': 1.0
                })
                print(f"{cf:.0f}Hz: {e1:>6.1f} {e2:>6.1f} {diff_db:>6.1f} -{reduction:.1f}dB")
                adjustments_made = True
            else:
                print(f"{cf:.0f}Hz: {e1:>6.1f} {e2:>6.1f} {diff_db:>6.1f} None")

    if valid_bands == 0:
        print("No valid frequency bands for analysis")
    elif not adjustments_made:
        print("No significant masking detected - no EQ adjustments needed")

    return eq_settings

def apply_eq_to_segment(audio_segment, eq_settings, sample_rate):
    """EQ settings auf audio segment"""
    if not eq_settings:
        return audio_segment

    samples = np.array(audio_segment.get_array_of_samples())
    if audio_segment.channels > 1:
        samples = samples.reshape((-1, audio_segment.channels))
        samples = samples.mean(axis=1)

    samples = samples.astype(np.float32)
    samples /= 32768.0  # Normalize for 16-bit audio

    for setting in eq_settings:
        b, a = signal.iirpeak(setting['freq'], Q=setting['q'], fs=sample_rate)
        samples = signal.lfilter(b, a, samples)

    samples = np.clip(samples * 32768.0, -32768, 32767).astype(np.int16)
    return AudioSegment(
        samples.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1
    )

def collect_sound_creation_data():
    """Collect all important data about sound creation for feedback"""
    global reverb_room_size, reverb_damping, reverb_wet_level, reverb_width
    global room_detected, room_size, damping, wet_level, width
    global reverb_enabled
    
    data = {
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'image_description': image_input_description[0] if image_input_description else "No description",
        'original_tags': {
            'scene_and_location': scene_and_location_tags if 'scene_and_location_tags' in globals() else [],
            'objects': recognized_tags,
            'importance_values': importance_values if 'importance_values' in globals() else []
        },
        'reverb_settings': {
            'enabled': reverb_enabled,
            'room_size': room_size if 'room_size' in globals() and room_size is not None else None,
            'damping': damping if 'damping' in globals() and damping is not None else None,
            'wet_level': wet_level if 'wet_level' in globals() and wet_level is not None else None,
            'width': width if 'width' in globals() and width is not None else None,
            'image_based_room_detected': room_detected if 'room_detected' in globals() else None
        },
        'processed_tracks': [],
        'settings': saved_sound_settings[-1] if saved_sound_settings else {},
        'console_log': ""
    }

    # Add track information
    for track in processed_tracks:
        data['processed_tracks'].append({
            'name': track['name'],
            'duration': len(track['audio']),
            'position': track['position'],
            'pan': track['pan'],
            'is_atmo': track['is_atmo']
        })

    return data

def save_console_log_to_file():
    """Save console logs to file for email attachment"""
    log_content = f"""
=== SOUND CREATION LOG ===
Timestamp: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Image: {image_input_description[0] if image_input_description else "No image"}

=== SETTINGS ===
{saved_sound_settings[-1] if saved_sound_settings else "No settings"}

=== PROCESSED TRACKS ===
"""
    for track in processed_tracks:
        log_content += f"- {track['name']}: {len(track['audio'])/1000:.1f}s, pos: {track['position']/1000:.1f}s, pan: {track['pan']}\n"

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    filename = os.path.join(log_dir, f"sound_creation_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(filename, 'w') as f:
        f.write(log_content)

    return filename

def send_feedback_email(rating, feedback_text):
    """Send feedback via email"""
    try:
        # Collect data
        creation_data = collect_sound_creation_data()
        log_file = save_console_log_to_file()

        # Create email
        msg = MIMEMultipart()
        msg['From'] = EMAIL_CONFIG['sender_email']
        msg['To'] = EMAIL_CONFIG['receiver_email']
        msg['Subject'] = f"Sound Creation Feedback - {creation_data['timestamp']}"

        # Email Body
        body = f"""
=== FEEDBACK FORM ===
Rating: {'★' * rating}{'☆' * (5-rating)} ({rating}/5)
Timestamp: {creation_data['timestamp']}

=== USER FEEDBACK ===
{feedback_text}

=== SOUND CREATION DATA ===
Image Description: {creation_data['image_description']}

Original Tags:
- Scene/Location: {', '.join(creation_data['original_tags']['scene_and_location'])}
- Objects: {', '.join(creation_data['original_tags']['objects'])}
- Importance Values: {creation_data['original_tags']['importance_values']}

Processed Tracks ({len(creation_data['processed_tracks'])}):
"""
        for track in creation_data['processed_tracks']:
            track_type = "Atmo" if track['is_atmo'] else "Object"
            body += f"- {track['name']} ({track_type}): {track['duration']/1000:.1f}s, pos: {track['position']/1000:.1f}s, pan: {track['pan']}\n"

        body += f"""
=== SETTINGS ===
Samplerate: {creation_data['settings'].get('samplerate', 'N/A')}
Duration: {creation_data['settings'].get('duration_min', 'N/A')}-{creation_data['settings'].get('duration_max', 'N/A')}s
License: {creation_data['settings'].get('license_type', 'N/A')}
Filetype: {creation_data['settings'].get('filetype', 'N/A')}
Quality Mode: {creation_data['settings'].get('quality_mode', 'N/A')}
Prefer Rating: {creation_data['settings'].get('prefer_rating', 'N/A')}
"""

        msg.attach(MIMEText(body, 'plain'))

        # Add log file as attachment
        with open(log_file, 'rb') as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f"attachment; filename= {log_file}")
            msg.attach(part)

        # Send email
        server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
        server.starttls()
        server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
        server.send_message(msg)
        server.quit()

        print(f"Feedback sent successfully to {EMAIL_CONFIG['receiver_email']}")
        
        # Clean up log file
        try:
            os.remove(log_file)
        except:
            pass
            
        return True

    except Exception as e:
        print(f"Error sending feedback: {e}")
        return False

# ----------------------------------------------------------------------
# AUDIO PROCESSING FUNCTIONS
# ----------------------------------------------------------------------

def create_bark_filterbank(sample_rate):
    """Create bark filterbank for audio analysis"""
    filters = []
    min_bandwidth = 50

    for i in range(len(BARK_BANDS)-1):
        low = BARK_BANDS[i]
        high = BARK_BANDS[i+1]

        if (high - low) < min_bandwidth or high >= sample_rate/2 * 0.95:
            continue

        if low < 100:
            if sample_rate < 48000:
                continue
            sos = signal.butter(2, [low, high], btype='bandpass',
                              fs=sample_rate, output='sos')
        else:
            sos = signal.butter(4, [low, high], btype='bandpass',
                              fs=sample_rate, output='sos')

        center_freq = (low + high) / 2
        filters.append((sos, center_freq))

    return filters

def analyze_bark_energy(audio_segment, filterbank, sample_rate):
    """Analyze audio energy across bark bands"""
    if not AUDIO_AVAILABLE:
        return []
        
    samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float64)
    if audio_segment.channels > 1:
        samples = samples.reshape((-1, audio_segment.channels)).mean(axis=1)

    max_val = np.max(np.abs(samples))
    if max_val > 0:
        samples /= max_val
        samples /= np.sqrt(np.mean(samples**2))

    sos_dc = signal.butter(4, 30, btype='highpass', fs=sample_rate, output='sos')
    samples = signal.sosfiltfilt(sos_dc, samples)

    energies = []
    for sos, center_freq in filterbank:
        try:
            filtered = signal.sosfiltfilt(sos, samples)

            if center_freq < 200:
                window = signal.windows.tukey(len(filtered), alpha=0.3)
                filtered = filtered * window

            chunk_size = min(len(filtered), int(sample_rate * (0.2 if center_freq < 200 else 0.05)))
            num_chunks = max(1, len(filtered) // chunk_size)

            chunk_energies = []
            for chunk in np.array_split(filtered, num_chunks):
                chunk_scaled = chunk / 4.0
                chunk_squared = chunk_scaled**2
                rms = np.sqrt(np.mean(chunk_squared)) * 4.0

                if rms > 0:
                    energy = 10 * np.log10(rms**2 + 1e-20)
                    chunk_energies.append(energy)

            if chunk_energies:
                energy = np.median(chunk_energies)
                if center_freq < 100:
                    energy = max(-200, min(-10, energy))
                energies.append((center_freq, energy))
            else:
                energies.append((center_freq, -200.0))

        except Exception as e:
            print(f"Error analyzing bark energy: {e}")
            energies.append((center_freq, -200.0))

    return energies

def calculate_energy_based_attenuation(audio_segment, filterbank, sample_rate, importance_value):
    """Calculate energy-based attenuation for audio normalization"""
    energies = analyze_bark_energy(audio_segment, filterbank, sample_rate)
    energy_values = [e for _, e in energies if e > -80]

    if not energy_values:
        return 0.0

    avg_energy = np.mean(energy_values)
    peak_energy = np.max(energy_values)
    energy_factor = min(1.0, max(0.0, (avg_energy + 60) / 40))
    energy_attenuation = -6.0 * energy_factor

    clamped_importance = max(0.0, min(1.0, importance_value))
    importance_weight = 0.7 + (clamped_importance * 0.3)
    combined_attenuation = energy_attenuation * (1.0 / importance_weight)

    combined_attenuation = max(-12.0, min(0.0, combined_attenuation))

    print(f"  Energy analysis: avg={avg_energy:.1f}dB, peak={peak_energy:.1f}dB")
    print(f"  Energy attenuation: {energy_attenuation:.1f}dB, "
          f"importance weight: {importance_weight:.2f}, "
          f"combined: {combined_attenuation:.1f}dB")

    return combined_attenuation

def download_and_process_sound(sound, original_tag, tag_instance_index, samplerate, 
                              importance_value=0.5, actual_samplerate=None, filterbank=None):
    """Download and process sound from FreeSound"""
    if not AUDIO_AVAILABLE:
        print("Audio processing libraries not available")
        return None
    
    # FIX: Ensure samplerate is integer
    if isinstance(samplerate, str):
        if samplerate == "any":
            samplerate = 44100
        elif samplerate.isdigit():
            samplerate = int(samplerate)
        else:
            samplerate = 44100
    
    if actual_samplerate is None:
        actual_samplerate = samplerate if isinstance(samplerate, (int, float)) else 44100
        

    # SICHEREN TAG-NAMEN erstellen (keine ungültigen Dateizeichen)
    safe_tag = re.sub(r'[<>:"/\\|?*]', '_', str(original_tag))
    
    # ABSOLUTEN PFAD mit Ordnerstruktur
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    sound_dir = os.path.join(base_dir, "downloaded_sounds")
    os.makedirs(sound_dir, exist_ok=True)
    
    filename = os.path.join(sound_dir, f"sound_{safe_tag}_{tag_instance_index}.mp3")
    
    print(f"Speichere Sound in: {filename}")

    try:
        # Sound herunterladen
        print(f"Lade Sound herunter: {sound['name']}")
        with requests.get(sound['previews']['preview-hq-mp3'], stream=True) as r:
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        print(f"Sound gespeichert: {filename}")
        
        # Sound laden
        audio = AudioSegment.from_file(filename, format="mp3")
        print(f"Sound geladen: {len(audio)}ms, {audio.channels} channels, {audio.frame_rate}Hz")

        if tag_instance_index == -1:
            processed_audio = audio
        else:
            processed_audio = audio.set_channels(1)

        if len(processed_audio) == 0 or processed_audio.dBFS == float('-inf'):
            print(f"File {filename} is too quiet or empty!")
            return None

        fade_duration = min(FADE_DURATION, len(processed_audio)//4)
        with_fades = processed_audio.fade_in(fade_duration).fade_out(fade_duration)

        if tag_instance_index == -1:
            target_dBFS = ATMO_LOUDNESS_DBFS
            gain_change = target_dBFS - with_fades.dBFS
            final_audio = with_fades.apply_gain(gain_change)
            print(f"→ Atmo sound normalized to {final_audio.dBFS:.1f} dBFS (fixed target: {ATMO_LOUDNESS_DBFS} dBFS)")
        else:
            clamped_importance = max(0.0, min(1.0, importance_value))

            energy_attenuation = calculate_energy_based_attenuation(
                with_fades,
                filterbank,
                actual_samplerate,
                clamped_importance
            )

            base_target_dBFS = MIN_LOUDNESS_DBFS + (MAX_LOUDNESS_DBFS - MIN_LOUDNESS_DBFS) * clamped_importance
            adjusted_target_dBFS = base_target_dBFS + energy_attenuation
            adjusted_target_dBFS = max(MIN_LOUDNESS_DBFS - 6.0, adjusted_target_dBFS)

            gain_change = adjusted_target_dBFS - with_fades.dBFS
            final_audio = with_fades.apply_gain(gain_change)

            print(f"→ Object sound '{sound['name']}' normalized to {final_audio.dBFS:.1f} dBFS")
            print(f"  Importance: {clamped_importance:.2f}, Base target: {base_target_dBFS:.1f} dBFS")
            print(f"  Energy attenuation: {energy_attenuation:.1f}dB, Final target: {adjusted_target_dBFS:.1f} dBFS")

        return final_audio

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        import traceback
        traceback.print_exc()
        return None

def search_sounds(api_key, tag, samplerate, duration_min, duration_max, 
                 license_type, filetype, min_rating=0, page=1):
    """Search for sounds on FreeSound"""
    time.sleep(1.5)  # Rate limiting

    filter_parts = [f'duration:[{duration_min} TO {duration_max}]']

    if samplerate != "any":
        filter_parts.append(f'samplerate:{samplerate}')
    if license_type != "any":
        filter_parts.append(f'license:"{license_type}"')
    if filetype != "any":
        filter_parts.append(f'type:"{filetype}"')
    if min_rating > 0:
        filter_parts.append(f'avg_rating:[{min_rating} TO 5]')

    filter_str = ' '.join(filter_parts)

    url = f'https://freesound.org/apiv2/search/text/?query={tag}&fields=id,name,previews,avg_rating,num_downloads,duration,description&token={api_key}&filter={filter_str}&page_size=10&page={page}'

    quality_mode = saved_sound_settings[0]['quality_mode'] if saved_sound_settings else True
    prefer_rating = saved_sound_settings[0]['prefer_rating'] if saved_sound_settings else True
    sort_param = 'rating_desc' if prefer_rating else 'downloads_desc'
    url += f'&sort={sort_param}'

    try:
        response = requests.get(url)
        if response.status_code == 200:
            results = response.json().get('results', [])
            return [r for r in results if r.get('avg_rating', 0) >= 2.0 and 'previews' in r]
        else:
            print(f"API returned status {response.status_code}")
            print(f"URL: {url}")
            return []
    except Exception as e:
        print(f"Search error for {tag}: {str(e)}")
        return []
    
def apply_reverb_to_audio(audio_segment, room_size_val, damping_val, wet_level_val, width_val):
    """Apply reverb to an AudioSegment using pedalboard"""
    if not AUDIO_AVAILABLE:
        print("Audio processing libraries not available")
        return audio_segment
        
    # Convert AudioSegment to numpy array for pedalboard
    samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
    if audio_segment.channels == 2:
        samples = samples.reshape((-1, 2)).T
    else:
        samples = samples.reshape((1, -1))

    # Normalize to [-1, 1]
    samples = samples / 32768.0

    # Create reverb pedalboard
    try:
        board = Pedalboard([
            Reverb(
                room_size=room_size_val,
                damping=damping_val,
                wet_level=wet_level_val,
                dry_level=1 - wet_level_val,
                width=width_val,
                freeze_mode=False
            )
        ])

        # Apply reverb
        sr = audio_segment.frame_rate
        processed_samples = board(samples, sr)

        # Convert back to AudioSegment
        if processed_samples.shape[0] == 2:  # stereo
            processed_samples = processed_samples.T.flatten()
        else:  # mono
            processed_samples = processed_samples.flatten()

        # Scale back to int16 range
        processed_samples = np.clip(processed_samples * 32768, -32768, 32767).astype(np.int16)

        # Create new AudioSegment
        return AudioSegment(
            processed_samples.tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=2 if audio_segment.channels == 2 else 1
        )
    except Exception as e:
        print(f"Error applying reverb: {e}")
        return audio_segment
    
def apply_reverb_to_mix():
    """Apply reverb to the current mix"""
    global current_mix_raw, current_mix_with_reverb, reverb_enabled
    global reverb_room_size, reverb_damping, reverb_wet_level, reverb_width
    
    # Note: This function needs to be called from within the ImageExtenderApp class
    # where it can access the current mix through the instance
    print("apply_reverb_to_mix should be called from class instance")
    return False

def remove_reverb_from_mix():
    """Remove reverb and return to raw mix"""
    global reverb_enabled
    
    if not reverb_enabled:
        print("Reverb is not currently applied")
        return
    
    reverb_enabled = False
    print("✅ Reverb removed")




# ----------------------------------------------------------------------
# GUI CLASSES
# ----------------------------------------------------------------------

class ImageExtenderApp:
    """Main application window"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Image Extender")
        
        # Modern color palette
        self.MODERN_BG = '#1e1e1e'  # Dark gray background
        self.MODERN_FG = '#ffffff'  # White text
        self.MODERN_ACCENT = '#4a9eff'  # Blue accent
        self.MODERN_SECONDARY = '#2d2d2d'  # Slightly lighter gray
        self.MODERN_HIGHLIGHT = '#00d4aa'  # Teal highlight
        self.MODERN_DANGER = '#ff4757'  # Red for important actions
        self.MODERN_SUCCESS = '#2ed573'  # Green for success
        
        #self.root.attributes('-fullscreen', True)
        #self.root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
        # Configure main window
        self.root.geometry("1400x900")
        self.root.configure(bg=self.MODERN_BG)
        
        # Center window on screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - 1400) // 2
        y = (screen_height - 900) // 2
        self.root.geometry(f"1400x900+{x}+{y}")
        
        # Configure styles
        style = ttk.Style()
        style.theme_use('clam')
        '''
        # Use modern theme if available
        if 'vista' in style.theme_names():
            style.theme_use('vista')
        elif 'alt' in style.theme_names():
            style.theme_use('alt')
        else:
            style.theme_use('clam')
        '''
        # Configure modern styles
        style.configure("Modern.TFrame", background=self.MODERN_BG)
        style.configure("Modern.TLabel", 
                        background=self.MODERN_BG, 
                        foreground=self.MODERN_FG,
                        font=("Segoe UI", 10))
        
        style.configure("Modern.TButton", 
                        background=self.MODERN_ACCENT,
                        foreground=self.MODERN_FG,
                        borderwidth=0,
                        focusthickness=0,
                        focuscolor='none',
                        font=("Segoe UI", 10),
                        padding=8)
        
        style.map("Modern.TButton",
                background=[('active', '#3a8eff')],
                foreground=[('active', self.MODERN_FG)])
        
        
        # 1. FIX TAB TEXT - Use ttk styling with explicit colors
        style.configure("TNotebook",
                        background=self.MODERN_BG,
                        borderwidth=0,
                        border=self.MODERN_SECONDARY)

        style.configure("TNotebook.Tab",
                        background=self.MODERN_SECONDARY,
                        foreground=self.MODERN_FG,
                        padding=[15, 8],
                        font=("Segoe UI", 11, "bold"),
                        lightcolor=self.MODERN_SECONDARY,
                        darkcolor=self.MODERN_SECONDARY,
                        bordercolor=self.MODERN_SECONDARY)

        style.map("TNotebook.Tab",
                background=[("selected", self.MODERN_ACCENT), ("active", self.MODERN_ACCENT)],
                foreground=[("selected", self.MODERN_FG), ("!selected", self.MODERN_FG)],
                focuscolor=[("selected", self.MODERN_ACCENT), ("!selected", self.MODERN_SECONDARY)])
        
        # 2. FIX COMBOBOX/DROPDOWN - This is the main issue
        # Configure the combobox entry field
        style.configure("TCombobox",
                        fieldbackground=self.MODERN_SECONDARY,  # Background of the text field
                        background=self.MODERN_SECONDARY,       # Background of the dropdown arrow
                        foreground=self.MODERN_FG,              # Text color
                        selectbackground=self.MODERN_ACCENT,    # Background when text is selected
                        selectforeground=self.MODERN_FG,        # Text color when selected
                        borderwidth=2,
                        relief="flat",
                        arrowcolor=self.MODERN_FG,
                        arrowsize=12)

        # Configure the dropdown list
        style.map("TCombobox",
                fieldbackground=[("readonly", self.MODERN_SECONDARY), ("disabled", self.MODERN_SECONDARY)],
                foreground=[("readonly", self.MODERN_FG), ("disabled", "#666666")],
                background=[("readonly", self.MODERN_SECONDARY), ("disabled", self.MODERN_SECONDARY)],
                selectbackground=[("readonly", self.MODERN_ACCENT)],
                selectforeground=[("readonly", self.MODERN_FG)],
                arrowcolor=[("readonly", self.MODERN_FG), ("disabled", "#666666")])
        
        # Configure scrollbars
        style.configure("Vertical.TScrollbar", 
                        background=self.MODERN_SECONDARY,
                        troughcolor=self.MODERN_BG,
                        borderwidth=0,
                        arrowcolor=self.MODERN_FG)
        
        # Create status bar
        self.status_bar = tk.Label(
            root, 
            text="Ready", 
            bd=1,
            relief=tk.FLAT,
            anchor=tk.W,
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_FG,
            font=('Segoe UI', 10),
            height=1,
            padx=15,
            pady=8
        )
        self.status_bar.pack(side=tk.TOP, fill=tk.X, pady=(0, 0))
        
        # Create header with title and restart button
        header_frame = tk.Frame(root, bg=self.MODERN_BG, height=60)
        header_frame.pack(fill=tk.X, pady=(10, 0))
        header_frame.pack_propagate(False)
        
        # Application title
        title_label = tk.Label(
            header_frame,
            text="IMAGE EXTENDER",
            bg=self.MODERN_BG,
            fg=self.MODERN_FG,
            font=("Segoe UI", 18, "bold")
        )
        title_label.pack(side=tk.LEFT, padx=20)
        
        # Restart button in header
        self.restart_button = tk.Button(
            header_frame,
            text="🔄 New Session",
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_FG,
            font=("Segoe UI", 10),
            command=self.restart_application,
            relief="flat",
            borderwidth=0,
            padx=15,
            pady=8,
            cursor="hand2",
            activebackground=self.MODERN_ACCENT,
            activeforeground=self.MODERN_FG
        )
        self.restart_button.pack(side=tk.RIGHT, padx=20)
        
        # Add subtle separator
        separator = ttk.Separator(root, orient=tk.HORIZONTAL)
        separator.pack(fill=tk.X, pady=(5, 5))
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        
        # Create tabs
        self.setup_tab1()  # Image Upload/Generation
        self.setup_tab2()  # Object Detection
        self.setup_tab3()  # Sound Settings
        self.setup_tab4()  # Download & Mix
        self.setup_tab5()  # Feedback
        
        self.set_default_sound_settings()
        
        # Modern color variables accessible throughout class
        self.colors = {
            'bg': self.MODERN_BG,
            'fg': self.MODERN_FG,
            'accent': self.MODERN_ACCENT,
            'secondary': self.MODERN_SECONDARY,
            'highlight': self.MODERN_HIGHLIGHT,
            'danger': self.MODERN_DANGER,
            'success': self.MODERN_SUCCESS
        }
         
    def update_status(self, message):
        """Update status bar message"""
        self.status_bar.config(text=message)
        self.root.update()
        
    def setup_tab1(self):
        """Setup Image Upload/Generation tab"""
        self.tab1 = tk.Frame(self.notebook, bg=self.MODERN_BG)
        self.notebook.add(self.tab1, text="Image")

        # Main container with padding
        main_container = tk.Frame(self.tab1, bg=self.MODERN_BG)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Headphones recommendation label - modern version
        headphones_label = tk.Label(
            main_container,
            text="Use headphones for optimal experience",
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_FG,
            font=("Segoe UI", 12),
            padx=20,
            pady=15,
            relief="flat",
            bd=0
        )
        headphones_label.pack(fill=tk.X, pady=(0, 20))
        
        # Mode selection frame - modern version
        mode_frame = tk.LabelFrame(
            main_container, 
            text="Image Source",
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_FG,
            font=("Segoe UI", 13, "bold"),
            relief="flat",
            bd=1,
            padx=15,
            pady=15
        )
        mode_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.mode_var = tk.StringVar(value="upload")
        
        # Modern radio buttons container
        radio_container = tk.Frame(mode_frame, bg=self.MODERN_SECONDARY)
        radio_container.pack(expand=True)
        
        # Upload Image radio button
        upload_radio = tk.Radiobutton(
            radio_container, 
            text="Upload Image", 
            variable=self.mode_var,
            value="upload", 
            command=self.update_image_mode,
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_FG,
            selectcolor=self.MODERN_ACCENT,
            font=("Segoe UI", 11),
            activebackground=self.MODERN_SECONDARY,
            activeforeground=self.MODERN_FG,
            cursor="hand2"
        )
        upload_radio.pack(side=tk.LEFT, padx=20, pady=5)
        
        # Generate Image radio button
        generate_radio = tk.Radiobutton(
            radio_container,
            text="Generate Image", 
            variable=self.mode_var,
            value="generate", 
            command=self.update_image_mode,
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_FG,
            selectcolor=self.MODERN_ACCENT,
            font=("Segoe UI", 11),
            activebackground=self.MODERN_SECONDARY,
            activeforeground=self.MODERN_FG,
            cursor="hand2"
        )
        generate_radio.pack(side=tk.LEFT, padx=20, pady=5)
        
        # Upload frame - modern version
        self.upload_frame = tk.Frame(
            main_container, 
            bg=self.MODERN_BG,
            highlightbackground=self.MODERN_SECONDARY,
            highlightthickness=1,
            highlightcolor=self.MODERN_SECONDARY
        )
        
        # Modern browse button
        browse_button = tk.Button(
            self.upload_frame,
            text="Browse...",
            command=self.upload_image,
            bg=self.MODERN_ACCENT,
            fg=self.MODERN_FG,
            font=("Segoe UI", 10),
            relief="flat",
            borderwidth=0,
            padx=20,
            pady=10,
            cursor="hand2",
            activebackground="#3a8eff",
            activeforeground=self.MODERN_FG
        )
        browse_button.pack(pady=20)
        
        # Modern image label
        self.image_label = tk.Label(
            self.upload_frame, 
            text="No image selected",
            bg=self.MODERN_BG,
            fg="#888888",  # Gray for placeholder
            font=("Segoe UI", 10),
            pady=10
        )
        self.image_label.pack(pady=10)
        
        # Generate frame - modern version
        self.generate_frame = tk.Frame(
            main_container, 
            bg=self.MODERN_BG,
            highlightbackground=self.MODERN_SECONDARY,
            highlightthickness=1,
            highlightcolor=self.MODERN_SECONDARY
        )
        
        # Prompt label
        prompt_label = tk.Label(
            self.generate_frame, 
            text="Prompt:",
            bg=self.MODERN_BG,
            fg=self.MODERN_FG,
            font=("Segoe UI", 11, "bold"),
            anchor=tk.W
        )
        prompt_label.pack(fill=tk.X, padx=15, pady=(15, 5))
        
        # Modern text entry with scrollbar
        text_frame = tk.Frame(self.generate_frame, bg=self.MODERN_BG)
        text_frame.pack(fill=tk.X, padx=15, pady=5)
        
        self.prompt_entry = tk.Text(
            text_frame, 
            height=4, 
            width=50,
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_FG,
            insertbackground=self.MODERN_FG,
            font=("Segoe UI", 10),
            relief="flat",
            wrap=tk.WORD,
            padx=10,
            pady=10
        )
        
        # Add scrollbar for long prompts
        text_scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.prompt_entry.yview)
        self.prompt_entry.configure(yscrollcommand=text_scrollbar.set)
        
        self.prompt_entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Insert placeholder text
        self.prompt_entry.insert("1.0", "Example: a realistic photo of a forest with trees and birds")
        self.prompt_entry.config(fg="#888888")  # Gray placeholder
        
        # Bind events for placeholder behavior
        def on_focus_in(event):
            if self.prompt_entry.get("1.0", "end-1c") == "Example: a realistic photo of a forest with trees and birds":
                self.prompt_entry.delete("1.0", tk.END)
                self.prompt_entry.config(fg=self.MODERN_FG)
        
        def on_focus_out(event):
            if not self.prompt_entry.get("1.0", "end-1c").strip():
                self.prompt_entry.insert("1.0", "Example: a realistic photo of a forest with trees and birds")
                self.prompt_entry.config(fg="#888888")
        
        self.prompt_entry.bind("<FocusIn>", on_focus_in)
        self.prompt_entry.bind("<FocusOut>", on_focus_out)
        
        # Modern generate button
        generate_button = tk.Button(
            self.generate_frame,
            text="Generate Image",
            command=self.generate_image,
            bg=self.MODERN_ACCENT,
            fg=self.MODERN_FG,
            font=("Segoe UI", 10),
            relief="flat",
            borderwidth=0,
            padx=25,
            pady=12,
            cursor="hand2",
            activebackground="#3a8eff",
            activeforeground=self.MODERN_FG
        )
        generate_button.pack(pady=15)
        
        # Modern generated image label
        self.generate_image_label = tk.Label(
            self.generate_frame, 
            text="No image generated yet",
            bg=self.MODERN_BG,
            fg="#888888",
            font=("Segoe UI", 10),
            pady=15
        )
        self.generate_image_label.pack(pady=10)
        
        # Initially show upload frame
        self.update_image_mode()
        
    def update_image_mode(self):
        """Update UI based on image mode selection"""
        mode = self.mode_var.get()
        
        # Hide both frames
        self.upload_frame.pack_forget()
        self.generate_frame.pack_forget()
        
        # Show selected frame with modern styling
        if mode == "upload":
            self.upload_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        else:
            self.generate_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
    def upload_image(self):
        # Set initial directory to images folder
        initial_dir = os.path.join(BASE_DIR, "images")
        if not os.path.exists(initial_dir):
            initial_dir = BASE_DIR
        
        # Modern file dialog would require custom implementation
        # For now, just use the default
        filepath = filedialog.askopenfilename(
            title="Select Image",
            initialdir=initial_dir,
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        
        if filepath:
            global IMAGE_FILE
            IMAGE_FILE = filepath
            
            # Display image with modern styling
            try:
                img = Image.open(filepath)
                img.thumbnail((400, 400))
                photo = ImageTk.PhotoImage(img)
                
                # Update label with modern styling
                self.image_label.config(
                    image=photo,
                    text="",
                    bg=self.MODERN_BG
                )
                self.image_label.image = photo
                
                self.update_status(f"Image loaded: {os.path.basename(filepath)}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load image: {e}")

            self.update_tab4_image()
                
    def generate_image(self):
        """Generate image from prompt"""
        if not OPENAI_AVAILABLE:
            messagebox.showwarning("OpenAI Required", 
                                "OpenAI library is not installed. Please install with: pip install openai")
            return
            
        if not openai_api_key:
            messagebox.showwarning("API Key Required", 
                                "Please set your OpenAI API key in the code")
            return
            
        prompt = self.prompt_entry.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showwarning("Prompt Required", "Please enter a prompt")
            return
            
        # Show loading
        self.update_status("Generating image...")
        
        try:
            # Enrich prompt for realism
            enriched_prompt = self.enrich_prompt_for_realism(prompt)
            
            client = openai.OpenAI(api_key=openai_api_key)

            # Generate image
            response = client.images.generate(
                #model="dall-e-2",
                model="gpt-image-1",
                prompt=enriched_prompt,
                size="1024x1024",
                quality="low",
                n=1,
                #response_format="b64_json",
            )
            
            # Decode and save image
            image_data = base64.b64decode(response.data[0].b64_json)
            global IMAGE_FILE
            IMAGE_FILE = "generated_image.png"

             # Save in images folder
            images_dir = os.path.join(BASE_DIR, "images")
            os.makedirs(images_dir, exist_ok=True)
            
            # Create unique filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_{timestamp}.png"
            IMAGE_FILE = os.path.join(images_dir, filename)
            
            with open(IMAGE_FILE, "wb") as f:
                f.write(image_data)
                
            # Display image in the generate frame
            try:
                img = Image.open(IMAGE_FILE)
                img.thumbnail((400, 400))
                photo = ImageTk.PhotoImage(img)
                
                # Update the generate frame image label
                self.generate_image_label.config(image=photo)
                self.generate_image_label.image = photo  # Keep a reference!
                self.generate_image_label.config(text="")
                
                self.update_status(f"Image generated: {IMAGE_FILE} - click on object detection tab")
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not display generated image: {e}")

            self.update_tab4_image()
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate image: {e}")
            self.update_status("Error generating image")
            
    def enrich_prompt_for_realism(self, user_prompt):
        """Use GPT to rewrite prompt for realistic photo"""
        client = openai.OpenAI(api_key=openai_api_key)

        try:
            completion = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "You are a prompt assistant for generating realistic images."},
                    {"role": "user", "content": (
                        f"Here is a user prompt: '{user_prompt}'. "
                        "Rewrite this prompt so it describes a realistic photo, "
                        "with the scene, location, and objects, avoiding drawings, cartoons, or illustrations."
                    )}
                ],
                temperature=0.7
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error enriching prompt: {e}")
            return user_prompt
            
    def setup_tab2(self):
        """Setup Object Detection tab"""
        self.tab2 = tk.Frame(self.notebook, bg=self.MODERN_BG)
        self.notebook.add(self.tab2, text="Object Detection")
        
        # Main container with padding
        main_container = tk.Frame(self.tab2, bg=self.MODERN_BG)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)
        
        # Settings frame - modern version
        settings_frame = tk.LabelFrame(
            main_container, 
            text="Detection Settings",
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_FG,
            font=("Segoe UI", 12, "bold"),
            relief="flat",
            bd=1,
            padx=15,
            pady=15
        )
        settings_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Configure grid for better alignment
        settings_frame.grid_columnconfigure(1, weight=1)
        
        # Method selection - modern version
        method_label = tk.Label(
            settings_frame, 
            text="Method:",
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_FG,
            font=("Segoe UI", 10)
        )
        method_label.grid(row=0, column=0, sticky=tk.W, pady=10, padx=(0, 10))
        
        self.method_var = tk.StringVar(value="OpenAI (GPT-4.1)")
        method_combo = ttk.Combobox(
            settings_frame, 
            textvariable=self.method_var,
            values=["Gemini (MediaPipe)", "OpenAI (GPT-4.1)"],
            state="readonly",
            width=25,
            font=("Segoe UI", 10)
        )
        method_combo.grid(row=0, column=1, sticky=tk.W, pady=10, padx=(0, 10))
        
        # Style the combobox
        style = ttk.Style()
        style.configure("TCombobox",
                        fieldbackground=self.MODERN_SECONDARY,
                        background=self.MODERN_SECONDARY,
                        foreground=self.MODERN_FG,
                        borderwidth=0,
                        relief="flat")
        
        # Add callback when method changes
        method_combo.bind('<<ComboboxSelected>>', lambda e: self.update_detection_method_ui())
        
        # Score threshold - modern version
        self.score_threshold_label = tk.Label(
            settings_frame, 
            text="Score Threshold:",
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_FG,
            font=("Segoe UI", 10)
        )
        self.score_threshold_label.grid(row=1, column=0, sticky=tk.W, pady=10, padx=(0, 10))
        
        self.score_threshold = tk.DoubleVar(value=0.5)
        
        # Slider container
        slider_frame1 = tk.Frame(settings_frame, bg=self.MODERN_SECONDARY)
        slider_frame1.grid(row=1, column=1, sticky=tk.W+tk.E, pady=10)
        
        self.score_threshold_slider = ttk.Scale(
            slider_frame1, 
            from_=0.0, 
            to=1.0, 
            variable=self.score_threshold,
            orient=tk.HORIZONTAL,
            length=250
        )
        self.score_threshold_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        self.score_threshold_value = tk.Label(
            slider_frame1,
            textvariable=self.score_threshold,
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_HIGHLIGHT,
            font=("Segoe UI", 10, "bold"),
            width=6
        )
        self.score_threshold_value.pack(side=tk.RIGHT)
        
        # Max results - modern version
        self.max_results_label = tk.Label(
            settings_frame, 
            text="Max Results:",
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_FG,
            font=("Segoe UI", 10)
        )
        self.max_results_label.grid(row=2, column=0, sticky=tk.W, pady=10, padx=(0, 10))
        
        self.max_results = tk.IntVar(value=5)
        
        # Slider container
        slider_frame2 = tk.Frame(settings_frame, bg=self.MODERN_SECONDARY)
        slider_frame2.grid(row=2, column=1, sticky=tk.W+tk.E, pady=10)
        
        self.max_results_slider = ttk.Scale(
            slider_frame2,
            from_=1, 
            to=50, 
            variable=self.max_results,
            orient=tk.HORIZONTAL,
            length=250
        )
        self.max_results_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        self.max_results_value = tk.Label(
            slider_frame2,
            textvariable=self.max_results,
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_HIGHLIGHT,
            font=("Segoe UI", 10, "bold"),
            width=6
        )
        self.max_results_value.pack(side=tk.RIGHT)
        
        # Run button - modern version
        run_button_frame = tk.Frame(settings_frame, bg=self.MODERN_SECONDARY)
        run_button_frame.grid(row=3, column=0, columnspan=2, pady=(15, 5), sticky=tk.EW)
        
        run_button = tk.Button(
            run_button_frame,
            text="Start Recognition",
            command=self.run_object_detection,
            bg=self.MODERN_ACCENT,
            fg=self.MODERN_FG,
            font=("Segoe UI", 11),
            relief="flat",
            borderwidth=0,
            padx=25,
            pady=12,
            cursor="hand2",
            activebackground="#3a8eff",
            activeforeground=self.MODERN_FG
        )
        run_button.pack()
        
        # Results frame - modern version
        results_frame = tk.LabelFrame(
            main_container, 
            text="Detection Results",
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_FG,
            font=("Segoe UI", 12, "bold"),
            relief="flat",
            bd=1,
            padx=15,
            pady=15
        )
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # Configure grid for results frame
        results_frame.grid_columnconfigure(0, weight=1)
        results_frame.grid_rowconfigure(0, weight=1)
        
        # Text area for results - modern version
        self.results_text = scrolledtext.ScrolledText(
            results_frame, 
            height=15,
            bg=self.MODERN_BG,
            fg=self.MODERN_FG,
            insertbackground=self.MODERN_FG,
            font=("Consolas", 9),
            relief="flat",
            wrap=tk.WORD,
            padx=10,
            pady=10
        )
        
        # Style the scrollbar
        style.configure("Vertical.TScrollbar",
                        background=self.MODERN_SECONDARY,
                        troughcolor=self.MODERN_BG,
                        borderwidth=0,
                        arrowcolor=self.MODERN_FG)
        
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Status frame at bottom
        status_frame = tk.Frame(main_container, bg=self.MODERN_BG, height=30)
        status_frame.pack(fill=tk.X, pady=(5, 0))
        status_frame.pack_propagate(False)
        
        self.detection_status_label = tk.Label(
            status_frame,
            text="Ready for object detection",
            bg=self.MODERN_BG,
            fg="#888888",
            font=("Segoe UI", 9),
            anchor=tk.W
        )
        self.detection_status_label.pack(fill=tk.X, padx=5, pady=5)

        self.update_detection_method_ui()

    def update_detection_method_ui(self):
        """Update UI based on selected detection method"""
        method = self.method_var.get()
        
        # Update label colors based on method
        if method == "Gemini (MediaPipe)":
            # Show all controls for MediaPipe
            self.score_threshold_label.config(fg=self.MODERN_FG)
            self.score_threshold_slider.state(['!disabled'])
            self.score_threshold_value.config(fg=self.MODERN_HIGHLIGHT)
            
            self.max_results_label.config(fg=self.MODERN_FG)
            self.max_results_slider.state(['!disabled'])
            self.max_results_value.config(fg=self.MODERN_HIGHLIGHT)
            
            # Update status
            if hasattr(self, 'detection_status_label'):
                self.detection_status_label.config(
                    text="Using Gemini (MediaPipe) - Adjust threshold and max results",
                    fg=self.MODERN_HIGHLIGHT
                )
        else:  # OpenAI
            # Hide threshold and max results controls
            self.score_threshold_label.config(fg="#666666")
            self.score_threshold_slider.state(['disabled'])
            self.score_threshold_value.config(fg="#666666")
            
            self.max_results_label.config(fg="#666666")
            self.max_results_slider.state(['disabled'])
            self.max_results_value.config(fg="#666666")
            
            # Update status
            if hasattr(self, 'detection_status_label'):
                self.detection_status_label.config(
                    text="Using OpenAI (GPT-4.1) - AI-powered detection",
                    fg=self.MODERN_HIGHLIGHT
                )
        
    def run_object_detection(self):
        """Run object detection on the image"""
        if not IMAGE_FILE or not os.path.exists(IMAGE_FILE):
            messagebox.showwarning("No Image", "Please load or generate an image first")
            return
            
        method = self.method_var.get()
        
        # Update status label
        if hasattr(self, 'detection_status_label'):
            self.detection_status_label.config(
                text=f"Running object detection using {method}...",
                fg=self.MODERN_ACCENT
            )
        
        # Clear results with modern styling
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Running object detection...\n")
        self.results_text.see(tk.END)
        
        # Run detection based on method
        if method == "Gemini (MediaPipe)":
            self.run_mediapipe_detection()
        else:
            self.run_openai_detection()
            
    def run_mediapipe_detection(self):
        """Run object detection using MediaPipe"""
        if not MEDIAPIPE_AVAILABLE:
            messagebox.showwarning("MediaPipe Required", 
                                 "MediaPipe is not installed. Please install with: pip install mediapipe")
            return
            
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            # Download model if needed
            model_path = "efficientdet.tflite"
            if not os.path.exists(model_path):
                self.update_status("Downloading model...")
                url = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite"
                response = requests.get(url)
                with open(model_path, 'wb') as f:
                    f.write(response.content)
                    
            # Create detector
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.ObjectDetectorOptions(
                base_options=base_options,
                score_threshold=self.score_threshold.get(),
                max_results=self.max_results.get()
            )
            detector = vision.ObjectDetector.create_from_options(options)
            
            # Detect objects
            image = mp.Image.create_from_file(IMAGE_FILE)
            detection_result = detector.detect(image)
            
            # Extract results
            global recognized_tags, sound_pannings
            recognized_tags, sound_pannings, recognized_tags_with_scores = [], [], []
            image_width = image.width
            
            for detection in detection_result.detections:
                bbox = detection.bounding_box
                center_x = bbox.origin_x + bbox.width / 2
                pan_value = (center_x / image_width) * 2 - 1
                for category in detection.categories:
                    recognized_tags.append(category.category_name)
                    sound_pannings.append(pan_value)
                    score = round(category.score * 100, 1)
                    recognized_tags_with_scores.append(f"{category.category_name} ({score}%)")
            
            # Display results
            self.results_text.insert(tk.END, "=== MediaPipe Detection Results ===\n\n")
            self.results_text.insert(tk.END, "Recognized objects:\n")
            for tag in recognized_tags_with_scores:
                self.results_text.insert(tk.END, f"  - {tag}\n")
                
            self.results_text.insert(tk.END, f"\nPanning values: {sound_pannings}\n")
            
            self.update_status("Object detection completed (MediaPipe) - click on sound creation tab")
            
        except Exception as e:
            messagebox.showerror("Error", f"MediaPipe detection failed: {e}")
            self.update_status("Error in object detection")
            
    def run_openai_detection(self):
        """Run object detection using OpenAI"""
        if not OPENAI_AVAILABLE:
            messagebox.showwarning("OpenAI Required", 
                                 "OpenAI library is not installed. Please install with: pip install openai")
            return
            
        if not openai_api_key:
            messagebox.showwarning("API Key Required", 
                                 "Please set your OpenAI API key in the code")
            return
            
        try:
            client = openai.OpenAI(api_key=openai_api_key)
            
            # Encode image
            image_b64 = encode_image_to_base64(IMAGE_FILE)
            
            # Prepare prompt
            prompt = """
            Analyze the provided image and return the result only as a valid Python dictionary.

            The dictionary must have exactly the following structure and key names:
            {
                "objects": ["...", "..."],
                "scene_and_location": ["..."],
                "image_input_description": ["..."],
                "panning": ["...", "..."],
                "importance": ["...", "..."],
                "room_detected": ["..."],
                "room_size": ["..."],
                "damping": ["..."],
                "wet_level": ["..."],
                "width": ["..."]
            }

            Guidelines:
            - "objects": list the main recognizable items, people, or animals. When animals are visible, identify them at the most specific level you can confidently determine (e.g., “pigeon,” “sparrow,” “seagull”) and avoid generic labels unless necessary.
            - "scene_and_location": describe the type of place and environment or possible region. Out of this info form one ideal entry for the dictionary for sound search of atmo recordings by adding atmo. Give only the one merged phrase.
            - "image_input_description": describe the image itself in a single sentence. This will be used later as a check for the sounds if they really match to the image
            - "panning": list the horizontal positions of the objects as numerical values between -1 and +1, where -1 = far left, 0 = center, +1 = far right. The order must match the order of "objects".
            - "importance": check the scale of the objects in the picture (scale compare all the objects, for example if a normally small object is bigger than a normally big object it is more in the foreground of the image) and analyze the importance in the soundscape representing the image as numerical values between 0 and 1. bigger/more present in the picture = higher score. The order must match the order of "objects".
            - note that all tags or words are getting used to find sounds on freesound.org, phrase everything semantically so it will get good sound results

            Room and acoustic estimation guidelines:
            - "room_detected": boolean indicating whether the image depicts a closed indoor room (true) or an outdoor/open environment (false).
            - "room_size": float between 0.0 and 1.0 estimating how large the room acoustically feels (small rooms → lower values, large halls → higher values).
            - "damping": float between 0.0 and 1.0 estimating high-frequency absorption based on visible materials (soft furnishings, carpets, curtains → higher values; hard reflective surfaces → lower values).
            - "wet_level": float between 0.0 and 1.0 estimating the natural reverberation amount in the space (dry rooms → lower values, echoic spaces → higher values).
            - "width": float between 0.0 and 1.0 estimating perceived stereo width based on room shape and openness.
            - All values that are meant to be single values (room_detected, room_size, damping, wet_level, width) must be provided as SINGLE VALUES, not lists.
            - Only "objects", "panning", "importance", "scene_and_location", and "image_input_description" should be lists.
            - "room_detected" should be a boolean: true or false (without quotes)

            Rules:
            - Return **only** the dictionary (no markdown, no explanation, no additional text).
            - If uncertain, make reasonable guesses — do not leave fields empty.
            - Always provide at least one entry per list.

        """

            # Call OpenAI
            response = client.responses.create(
                model="gpt-4.1-mini",
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}"}
                    ]
                }],
                max_output_tokens=400,
            )

            output_text = response.output_text.strip()
            data = parse_gpt_output(output_text)
            
            # Store results globally
            global recognized_tags, sound_pannings, scene_and_location_tags
            global image_input_description, importance_values, room_detected
            global room_size, damping, wet_level, width
            
            recognized_tags = data.get("objects", [])
            sound_pannings = data.get("panning", [])
            scene_and_location_tags = data.get("scene_and_location", [])
            image_input_description = data.get("image_input_description", [])
            importance_values = data.get("importance", [])
            room_detected = data.get("room_detected", [])
            room_size = data.get("room_size", [])
            damping = data.get("damping", [])
            wet_level = data.get("wet_level", [])
            width = data.get("width", [])
            
            # Display results
            self.results_text.insert(tk.END, "=== OpenAI Detection Results ===\n\n")
            self.results_text.insert(tk.END, "Recognized objects:\n")
            for obj in recognized_tags:
                self.results_text.insert(tk.END, f"  - {obj}\n")
                
            self.results_text.insert(tk.END, f"\nPanning values: {sound_pannings}\n")
            
            if scene_and_location_tags:
                self.results_text.insert(tk.END, f"\nScene & location: {scene_and_location_tags}\n")
                
            if room_detected:
                self.results_text.insert(tk.END, f"\nRoom detected: {room_detected}\n")
                self.results_text.insert(tk.END, f"Room size: {room_size}\n")
                self.results_text.insert(tk.END, f"Damping: {damping}\n")
                self.results_text.insert(tk.END, f"Wet level: {wet_level}\n")
                self.results_text.insert(tk.END, f"Width: {width}\n")
                
            self.update_status("Object detection completed (OpenAI) - click on sound creation tab tab")
            
        except Exception as e:
            messagebox.showerror("Error", f"OpenAI detection failed: {e}")
            self.update_status("Error in object detection")
            
    def setup_tab3(self):
        """Setup Sound Settings tab"""
        self.tab3 = tk.Frame(self.notebook, bg=self.MODERN_BG)
        self.notebook.add(self.tab3, text="Sound Settings")
        
        # Main container
        main_container = tk.Frame(self.tab3, bg=self.MODERN_BG)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)
        
        # Settings container - modern version
        settings_frame = tk.LabelFrame(
            main_container,
            text="Sound Search Settings",
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_FG,
            font=("Segoe UI", 12, "bold"),
            relief="flat",
            bd=1,
            padx=15,
            pady=15
        )
        settings_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a canvas and scrollbar with modern styling
        canvas = tk.Canvas(
            settings_frame,
            bg=self.MODERN_SECONDARY,
            highlightthickness=0
        )
        
        scrollbar = ttk.Scrollbar(
            settings_frame,
            orient=tk.VERTICAL,
            command=canvas.yview,
            style="Vertical.TScrollbar"
        )
        
        scrollable_frame = tk.Frame(canvas, bg=self.MODERN_SECONDARY)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Title and description
        title_label = tk.Label(
            scrollable_frame,
            text="Configure Sound Search Parameters",
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_FG,
            font=("Segoe UI", 13, "bold"),
            pady=10
        )
        title_label.pack(fill=tk.X, pady=(0, 15))
        
        desc_label = tk.Label(
            scrollable_frame,
            text="Adjust these settings to control how sounds are searched and downloaded",
            bg=self.MODERN_SECONDARY,
            fg="#aaaaaa",
            font=("Segoe UI", 10),
            wraplength=600,
            justify=tk.LEFT
        )
        desc_label.pack(fill=tk.X, pady=(0, 20))
        
        # Quality mode toggle - modern version
        quality_frame = tk.Frame(scrollable_frame, bg=self.MODERN_SECONDARY)
        quality_frame.pack(fill=tk.X, pady=12, padx=10)
        
        quality_icon = tk.Label(
            quality_frame,
            text="⚡",
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_HIGHLIGHT,
            font=("Segoe UI", 14)
        )
        quality_icon.pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Label(
            quality_frame,
            text="Quality Mode:",
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_FG,
            font=("Segoe UI", 11)
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        self.quality_mode_var = tk.BooleanVar(value=True)
        
        # Modern toggle switch style
        quality_check = tk.Checkbutton(
            quality_frame,
            variable=self.quality_mode_var,
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_FG,
            selectcolor=self.MODERN_ACCENT,
            activebackground=self.MODERN_SECONDARY,
            activeforeground=self.MODERN_FG,
            cursor="hand2"
        )
        quality_check.pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Label(
            quality_frame,
            text="Prioritize high-quality sounds",
            bg=self.MODERN_SECONDARY,
            fg="#aaaaaa",
            font=("Segoe UI", 10)
        ).pack(side=tk.LEFT)
        
        # Prefer rating toggle - modern version
        rating_frame = tk.Frame(scrollable_frame, bg=self.MODERN_SECONDARY)
        rating_frame.pack(fill=tk.X, pady=12, padx=10)
        
        rating_icon = tk.Label(
            rating_frame,
            text="*",
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_HIGHLIGHT,
            font=("Segoe UI", 14)
        )
        rating_icon.pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Label(
            rating_frame,
            text="Preference:",
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_FG,
            font=("Segoe UI", 11)
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        self.prefer_rating_var = tk.BooleanVar(value=True)
        
        rating_check = tk.Checkbutton(
            rating_frame,
            variable=self.prefer_rating_var,
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_FG,
            selectcolor=self.MODERN_ACCENT,
            activebackground=self.MODERN_SECONDARY,
            activeforeground=self.MODERN_FG,
            cursor="hand2"
        )
        rating_check.pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Label(
            rating_frame,
            text="Rating over download count",
            bg=self.MODERN_SECONDARY,
            fg="#aaaaaa",
            font=("Segoe UI", 10)
        ).pack(side=tk.LEFT)
        
        # Add separator
        separator1 = ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL)
        separator1.pack(fill=tk.X, pady=20, padx=20)
        
        # Samplerate - modern version
        samplerate_frame = tk.Frame(scrollable_frame, bg=self.MODERN_SECONDARY)
        samplerate_frame.pack(fill=tk.X, pady=12, padx=10)
        
        samplerate_icon = tk.Label(
            samplerate_frame,
            text="Audio",
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_HIGHLIGHT,
            font=("Segoe UI", 14)
        )
        samplerate_icon.pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Label(
            samplerate_frame,
            text="Sample Rate:",
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_FG,
            font=("Segoe UI", 11),
            width=15,
            anchor=tk.W
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        self.samplerate_var = tk.StringVar(value="any")
        
        samplerate_combo = ttk.Combobox(
            samplerate_frame,
            textvariable=self.samplerate_var,
            values=["any", 8000, 11025, 22050, 44100, 48000, 96000],
            state="readonly",
            width=12,
            font=("Segoe UI", 10)
        )
        samplerate_combo.pack(side=tk.LEFT)
        
        # Duration - modern version
        duration_frame = tk.Frame(scrollable_frame, bg=self.MODERN_SECONDARY)
        duration_frame.pack(fill=tk.X, pady=12, padx=10)
        
        duration_icon = tk.Label(
            duration_frame,
            text="⏱️",
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_HIGHLIGHT,
            font=("Segoe UI", 14)
        )
        duration_icon.pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Label(
            duration_frame,
            text="Duration (seconds):",
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_FG,
            font=("Segoe UI", 11),
            width=15,
            anchor=tk.W
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        self.duration_min_var = tk.IntVar(value=2)
        
        duration_min_spin = ttk.Spinbox(
            duration_frame,
            from_=1,
            to=10,
            textvariable=self.duration_min_var,
            width=8,
            font=("Segoe UI", 10)
        )
        duration_min_spin.pack(side=tk.LEFT, padx=(0, 5))
        
        tk.Label(
            duration_frame,
            text="to",
            bg=self.MODERN_SECONDARY,
            fg="#aaaaaa",
            font=("Segoe UI", 10)
        ).pack(side=tk.LEFT, padx=5)
        
        self.duration_max_var = tk.IntVar(value=10)
        
        duration_max_spin = ttk.Spinbox(
            duration_frame,
            from_=5,
            to=200,
            textvariable=self.duration_max_var,
            width=8,
            font=("Segoe UI", 10)
        )
        duration_max_spin.pack(side=tk.LEFT, padx=(5, 0))
        
        # License type - modern version
        license_frame = tk.Frame(scrollable_frame, bg=self.MODERN_SECONDARY)
        license_frame.pack(fill=tk.X, pady=12, padx=10)
        
        license_icon = tk.Label(
            license_frame,
            text="📄",
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_HIGHLIGHT,
            font=("Segoe UI", 14)
        )
        license_icon.pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Label(
            license_frame,
            text="License Type:",
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_FG,
            font=("Segoe UI", 11),
            width=15,
            anchor=tk.W
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        self.license_var = tk.StringVar(value="any")
        
        license_combo = ttk.Combobox(
            license_frame,
            textvariable=self.license_var,
            values=["any", "Attribution", "Attribution NonCommercial", "Creative Commons 0"],
            state="readonly",
            width=25,
            font=("Segoe UI", 10)
        )
        license_combo.pack(side=tk.LEFT)
        
        # File type - modern version
        filetype_frame = tk.Frame(scrollable_frame, bg=self.MODERN_SECONDARY)
        filetype_frame.pack(fill=tk.X, pady=12, padx=10)
        
        filetype_icon = tk.Label(
            filetype_frame,
            text="File",
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_HIGHLIGHT,
            font=("Segoe UI", 14)
        )
        filetype_icon.pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Label(
            filetype_frame,
            text="File Type:",
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_FG,
            font=("Segoe UI", 11),
            width=15,
            anchor=tk.W
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        self.filetype_var = tk.StringVar(value="any")
        
        filetype_combo = ttk.Combobox(
            filetype_frame,
            textvariable=self.filetype_var,
            values=["any", "wav", "aiff", "ogg", "mp3", "m4a", "flac"],
            state="readonly",
            width=12,
            font=("Segoe UI", 10)
        )
        filetype_combo.pack(side=tk.LEFT)
        
        # Add separator before save button
        separator2 = ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL)
        separator2.pack(fill=tk.X, pady=30, padx=20)
        
        # Save button container
        save_button_container = tk.Frame(scrollable_frame, bg=self.MODERN_SECONDARY)
        save_button_container.pack(pady=10)
        
        # Modern save button
        save_button = tk.Button(
            save_button_container,
            text="Save Settings",
            command=self.save_sound_settings,
            bg=self.MODERN_ACCENT,
            fg=self.MODERN_FG,
            font=("Segoe UI", 11, "bold"),
            relief="flat",
            borderwidth=0,
            padx=30,
            pady=14,
            cursor="hand2",
            activebackground="#3a8eff",
            activeforeground=self.MODERN_FG
        )
        save_button.pack()
        
        # Status label
        self.settings_status_label = tk.Label(
            save_button_container,
            text="Settings will be saved automatically",
            bg=self.MODERN_SECONDARY,
            fg="#888888",
            font=("Segoe UI", 9),
            pady=5
        )
        self.settings_status_label.pack()
        
        # Pack canvas and scrollbar
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add some bottom padding
        bottom_padding = tk.Frame(main_container, height=10, bg=self.MODERN_BG)
        bottom_padding.pack(fill=tk.X)
        
    def save_sound_settings(self):
        """Save sound settings to global variable"""
        global saved_sound_settings, quality_mode, prefer_rating
        
        settings = {
            'samplerate': self.samplerate_var.get(),
            'duration_min': self.duration_min_var.get(),
            'duration_max': self.duration_max_var.get(),
            'license_type': self.license_var.get(),
            'filetype': self.filetype_var.get(),
            'quality_mode': self.quality_mode_var.get(),
            'prefer_rating': self.prefer_rating_var.get()
        }
        
        saved_sound_settings.append(settings)
        quality_mode = self.quality_mode_var.get()
        prefer_rating = self.prefer_rating_var.get()
        
        # Show modern feedback
        if hasattr(self, 'settings_status_label'):
            self.settings_status_label.config(
                text="✓ Settings saved successfully!",
                fg=self.MODERN_SUCCESS
            )
        
        # Schedule reset of status message
        if hasattr(self, 'root'):
            self.root.after(3000, lambda: 
                self.settings_status_label.config(
                    text="Settings will be saved automatically",
                    fg="#888888"
                ) if hasattr(self, 'settings_status_label') else None
            )
        
        self.update_status("Sound settings saved - ready for sound creation")
        
    def setup_tab4(self):
        """Setup Download & Mix tab - FIXED VERSION with scrollable right side"""
        self.tab4 = tk.Frame(self.notebook, bg=self.MODERN_BG)
        self.notebook.add(self.tab4, text="Sound Creation")
        
        # Main container with modern styling
        main_container = tk.Frame(self.tab4, bg=self.MODERN_BG)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)
        
        # Split layout container
        split_container = tk.Frame(main_container, bg=self.MODERN_BG)
        split_container.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights for better layout
        split_container.grid_columnconfigure(0, weight=0)  # Left column (image) - fixed width
        split_container.grid_columnconfigure(1, weight=1)  # Right column (controls) - expands
        split_container.grid_rowconfigure(0, weight=1)     # Main content row
        
        # Left frame for image (fixed width)
        left_frame = tk.Frame(
            split_container, 
            bg=self.MODERN_BG,
            highlightbackground=self.MODERN_SECONDARY,
            highlightthickness=1,
            width=320,  # Fixed width for image
            height=600
        )
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 15))
        left_frame.grid_propagate(False)  # Don't let children resize
        
        # Image title
        image_title = tk.Label(
            left_frame,
            text="Current Image",
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_FG,
            font=("Segoe UI", 12, "bold"),
            padx=15,
            pady=10
        )
        image_title.pack(fill=tk.X, pady=(0, 10))
        
        # Image display container with fixed size
        image_container = tk.Frame(
            left_frame,
            bg=self.MODERN_BG,
            relief="flat",
            padx=15,
            pady=15,
            width=290,
            height=290
        )
        image_container.pack_propagate(False)  # Prevent shrinking
        image_container.pack()
        
        # Modern image display label
        self.tab4_image_label = tk.Label(
            image_container, 
            text="No image loaded\n\nLoad or generate an image first",
            bg=self.MODERN_SECONDARY,
            fg="#888888",
            font=("Segoe UI", 10),
            relief="flat",
            anchor=tk.CENTER,
            padx=20,
            pady=30
        )
        self.tab4_image_label.pack(fill=tk.BOTH, expand=True)
        
        # ============================================================
        # RIGHT SIDE - FIXED TO FILL ALL AVAILABLE SPACE
        # ============================================================
        
        # Right container frame - FILLS ALL AVAILABLE SPACE
        right_container = tk.Frame(
            split_container, 
            bg=self.MODERN_BG
        )
        right_container.grid(row=0, column=1, sticky="nsew", padx=(0, 0))
        
        # Configure right container to use ALL available space
        right_container.grid_columnconfigure(0, weight=1)
        right_container.grid_rowconfigure(0, weight=1)
        
        # Create canvas and scrollbar that FILL the right container
        right_canvas = tk.Canvas(
            right_container,
            bg=self.MODERN_BG,
            highlightthickness=0
        )
        
        right_scrollbar = ttk.Scrollbar(
            right_container,
            orient=tk.VERTICAL,
            command=right_canvas.yview
        )
        
        # Scrollable frame for ALL right-side content
        scrollable_right_frame = tk.Frame(right_canvas, bg=self.MODERN_BG)
        
        # Configure canvas scrolling
        scrollable_right_frame.bind(
            "<Configure>",
            lambda e: right_canvas.configure(scrollregion=right_canvas.bbox("all"))
        )
        
        # Create window in canvas
        right_canvas.create_window((0, 0), window=scrollable_right_frame, anchor="nw", width=right_canvas.winfo_width())
        right_canvas.configure(yscrollcommand=right_scrollbar.set)
        
        # Make the canvas window expand horizontally
        def configure_canvas_window(event):
            # Update the canvas window width to fill the canvas
            right_canvas.itemconfig(1, width=event.width)  # 1 is the tag for the first item (our window)
        
        right_canvas.bind('<Configure>', configure_canvas_window)
        
        # ============================================================
        # ALL RIGHT-SIDE CONTENT GOES IN scrollable_right_frame
        # ============================================================
        
        # Main action button - top of scrollable area
        action_container = tk.Frame(scrollable_right_frame, bg=self.MODERN_BG)
        action_container.pack(fill=tk.X, pady=(0, 15))
        
        self.download_button = tk.Button(
            action_container,
            text="Download & Create Soundscape",
            command=self.download_and_mix_sounds,
            bg=self.MODERN_ACCENT,
            fg=self.MODERN_FG,
            font=("Segoe UI", 12, "bold"),
            relief="flat",
            borderwidth=0,
            padx=30,
            pady=16,
            cursor="hand2",
            activebackground="#3a8eff",
            activeforeground=self.MODERN_FG
        )
        self.download_button.pack(fill=tk.X)
        
        # Processing status indicator
        self.processing_indicator = tk.Label(
            action_container,
            text="",
            bg=self.MODERN_BG,
            fg="#888888",
            font=("Segoe UI", 9),
            pady=5
        )
        self.processing_indicator.pack(fill=tk.X)
        
        # Processing Log frame
        text_frame = tk.LabelFrame(
            scrollable_right_frame, 
            text="Processing Log",
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_FG,
            font=("Segoe UI", 12, "bold"),
            relief="flat",
            bd=1,
            padx=15,
            pady=15
        )
        text_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Modern text area with scrollbar
        text_container = tk.Frame(text_frame, bg=self.MODERN_SECONDARY)
        text_container.pack(fill=tk.BOTH, expand=True)
        
        self.daw_text = scrolledtext.ScrolledText(
            text_container,
            height=8,
            bg=self.MODERN_BG,
            fg=self.MODERN_FG,
            insertbackground=self.MODERN_FG,
            font=("Consolas", 9),
            relief="flat",
            wrap=tk.WORD,
            padx=12,
            pady=12
        )
        self.daw_text.pack(fill=tk.BOTH, expand=True)
        
        # Initial message
        self.daw_text.insert(tk.END, "Ready to create soundscape...\n")
        self.daw_text.insert(tk.END, "1. Load or generate an image\n")
        self.daw_text.insert(tk.END, "2. Run object detection\n")
        self.daw_text.insert(tk.END, "3. Click 'Download & Create Soundscape' above\n\n")
        self.daw_text.config(state=tk.DISABLED)
        
        def enable_text():
            self.daw_text.config(state=tk.NORMAL)
        
        def disable_text():
            self.daw_text.config(state=tk.DISABLED)
        
        # Store these functions for later use
        self.enable_log_text = enable_text
        self.disable_log_text = disable_text
        
        # DAW Controls frame - SCROLLABLE
        self.daw_controls_frame = tk.LabelFrame(
            scrollable_right_frame, 
            text="Mix Controls",
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_FG,
            font=("Segoe UI", 12, "bold"),
            relief="flat",
            bd=1,
            padx=15,
            pady=15
        )
        self.daw_controls_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Fallback method settings - top section
        settings_container = tk.Frame(self.daw_controls_frame, bg=self.MODERN_SECONDARY)
        settings_container.pack(fill=tk.X, pady=(0, 10))
        
        fallback_frame = tk.Frame(settings_container, bg=self.MODERN_SECONDARY)
        fallback_frame.pack(anchor=tk.W, pady=5)
        
        tk.Label(
            fallback_frame,
            text="Fallback Method:",
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_FG,
            font=("Segoe UI", 10)
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        self.fallback_method_var = tk.StringVar(value="OpenAI")
        
        fallback_combo = ttk.Combobox(
            fallback_frame,
            textvariable=self.fallback_method_var,
            values=["CSV", "OpenAI"],
            state="readonly",
            width=12,
            font=("Segoe UI", 10)
        )
        fallback_combo.pack(side=tk.LEFT)
        
        # Help text
        help_label = tk.Label(
            settings_container,
            text="Adjust track levels then click 'Refresh Mix' to apply changes",
            bg=self.MODERN_SECONDARY,
            fg="#888888",
            font=("Segoe UI", 9),
            anchor=tk.W
        )
        help_label.pack(fill=tk.X, pady=(5, 0))
        
        # DAW controls container - This will hold the track sliders
        self.daw_controls_container = tk.Frame(
            self.daw_controls_frame, 
            bg=self.MODERN_SECONDARY
        )
        self.daw_controls_container.pack(fill=tk.X, pady=(10, 0))
        
        # Initial placeholder for DAW controls
        placeholder_frame = tk.Frame(self.daw_controls_container, bg=self.MODERN_SECONDARY)
        placeholder_frame.pack(fill=tk.X, pady=20)
        
        placeholder_label = tk.Label(
            placeholder_frame,
            text="No tracks available yet\n\nProcess an image to create soundscape",
            bg=self.MODERN_SECONDARY,
            fg="#666666",
            font=("Segoe UI", 11),
            justify=tk.CENTER
        )
        placeholder_label.pack()
        
        # Playback controls container - ALWAYS VISIBLE AT BOTTOM
        playback_container = tk.Frame(self.daw_controls_frame, bg=self.MODERN_SECONDARY)
        playback_container.pack(fill=tk.X, pady=(15, 5))
        
        # Add separator
        separator = ttk.Separator(playback_container, orient=tk.HORIZONTAL)
        separator.pack(fill=tk.X, pady=(0, 10))
        
        # Playback buttons row
        button_row = tk.Frame(playback_container, bg=self.MODERN_SECONDARY)
        button_row.pack(fill=tk.X, pady=5)
        
        play_btn = tk.Button(
            button_row,
            text="▶ Play Mix",
            command=self._play_mix,
            bg=self.MODERN_ACCENT,
            fg=self.MODERN_FG,
            font=("Segoe UI", 10),
            relief="flat",
            borderwidth=0,
            padx=15,
            pady=8,
            cursor="hand2",
            activebackground="#3a8eff",
            activeforeground=self.MODERN_FG
        )
        play_btn.pack(side=tk.LEFT, padx=5)
        
        stop_btn = tk.Button(
            button_row,
            text="⏹ Stop",
            command=self._stop_mix,
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_FG,
            font=("Segoe UI", 10),
            relief="flat",
            borderwidth=0,
            padx=15,
            pady=8,
            cursor="hand2",
            activebackground="#333333",
            activeforeground=self.MODERN_FG
        )
        stop_btn.pack(side=tk.LEFT, padx=5)
        
        export_btn = tk.Button(
            button_row,
            text="Export Mix",
            command=self._export_mix,
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_FG,
            font=("Segoe UI", 10),
            relief="flat",
            borderwidth=0,
            padx=15,
            pady=8,
            cursor="hand2",
            activebackground="#333333",
            activeforeground=self.MODERN_FG
        )
        export_btn.pack(side=tk.LEFT, padx=5)
        
        refresh_btn = tk.Button(
            button_row,
            text="🔁 Refresh Mix",
            command=self._refresh_mix,
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_FG,
            font=("Segoe UI", 10),
            relief="flat",
            borderwidth=0,
            padx=15,
            pady=8,
            cursor="hand2",
            activebackground="#333333",
            activeforeground=self.MODERN_FG
        )
        refresh_btn.pack(side=tk.LEFT, padx=5)
        
        # Reverb buttons row (if room detected)
        self.reverb_container = tk.Frame(playback_container, bg=self.MODERN_SECONDARY)
        self.reverb_container.pack(fill=tk.X, pady=(10, 0))
        
        # Status label at bottom
        self.daw_status_label = tk.Label(
            playback_container,
            text="Ready",
            bg=self.MODERN_SECONDARY,
            fg="#888888",
            font=("Segoe UI", 9),
            anchor=tk.W,
            padx=5,
            pady=8
        )
        self.daw_status_label.pack(fill=tk.X, pady=(10, 0))
        
        # ============================================================
        # PACK THE CANVAS AND SCROLLBAR TO FILL ALL SPACE
        # ============================================================
        
        # Use grid instead of pack for better control
        right_canvas.grid(row=0, column=0, sticky="nsew")
        right_scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Configure mouse wheel scrolling
        def _on_mousewheel(event):
            right_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        # Bind mouse wheel to canvas
        right_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Update bind to include canvas width
        right_canvas.bind('<Configure>', lambda e: right_canvas.itemconfig(1, width=e.width))
        
        # Load image if available
        self.update_tab4_image()
        
        # Store reference for later
        self.right_canvas = right_canvas
        self.scrollable_right_frame = scrollable_right_frame
        
        # Force update to ensure proper layout
        self.tab4.update_idletasks()

    def set_default_sound_settings(self):
        """Set preset sound settings for exhibition"""
        global saved_sound_settings, quality_mode, prefer_rating
        
        preset_settings = {
            'samplerate': 'any',
            'duration_min': 2,
            'duration_max': 100,
            'license_type': 'any',
            'filetype': 'any',
            'quality_mode': True,
            'prefer_rating': True
        }
        
        saved_sound_settings = [preset_settings]
        quality_mode = True
        prefer_rating = True
        
        print("✓ Exhibition preset sound settings loaded")
        

    def download_and_mix_sounds(self):
        """Main function to download and mix sounds - Complete implementation"""
        global processed_tracks, used_sound_ids, downloaded_files
        
        # Clear previous tracks
        processed_tracks.clear()
        
        # Check if image was processed
        if not recognized_tags:
            messagebox.showwarning("No Objects Detected", 
                            "Please run object detection first in the Object Detection tab.")
            return
        
        # TEMPORARILY ALLOW BACKGROUND MUSIC during processing
        global current_active_tab, background_music_target_volume
        was_music_paused = (current_active_tab == "Sound Creation")
        
        if was_music_paused:
            # Update log
            if hasattr(self, 'daw_text') and self.daw_text.winfo_exists():
                self.enable_log_text()
                self.daw_text.insert(tk.END, "Temporarily resuming background music for processing...\n")
                self.daw_text.see(tk.END)
            
            background_music_target_volume = background_music_volume
            # Give music time to fade back in
            time.sleep(0.5)
        
        # Update UI state
        self.download_button.config(state=tk.DISABLED)
        
        # Update processing indicator
        if hasattr(self, 'processing_indicator'):
            self.processing_indicator.config(
                text="🔄 Processing sounds... (This may take 2-5 minutes)",
                fg=self.MODERN_ACCENT
            )
        
        # Use preset settings for exhibition
        if not saved_sound_settings:
            # Auto-load default settings for exhibition
            self.set_default_sound_settings()
        
        # Clear text area and show modern loading
        if hasattr(self, 'daw_text') and self.daw_text.winfo_exists():
            self.enable_log_text()
            self.daw_text.delete(1.0, tk.END)
            
            # Modern loading header
            self.daw_text.insert(tk.END, "=" * 60 + "\n")
            self.daw_text.insert(tk.END, "  SOUNDSCAPE CREATION PROCESS\n")
            self.daw_text.insert(tk.END, "=" * 60 + "\n\n")
            
            # Status message with icon
            self.daw_text.insert(tk.END, "🔄 Starting sound processing...\n")
            self.daw_text.insert(tk.END, f"Image: {image_input_description[0] if image_input_description else 'No description'}\n")
            self.daw_text.insert(tk.END, f"Objects detected: {len(recognized_tags)}\n\n")
            self.daw_text.see(tk.END)
        
        self.update_status("Starting sound processing... [this step might take 2 to 5 minutes]")
        
        # Get settings
        settings = saved_sound_settings[-1]
        samplerate = settings['samplerate']
        duration_min = settings['duration_min']
        duration_max = settings['duration_max']
        license_type = settings['license_type']
        filetype = settings['filetype']
        quality_mode = settings['quality_mode']
        prefer_rating = settings['prefer_rating']
        
        # Get fallback method
        fallback_method = self.fallback_method_var.get()
        
        # Update log with settings
        if hasattr(self, 'daw_text') and self.daw_text.winfo_exists():
            self.daw_text.insert(tk.END, "Using settings:\n")
            self.daw_text.insert(tk.END, f"  • Sample Rate: {samplerate}\n")
            self.daw_text.insert(tk.END, f"  • Duration: {duration_min}-{duration_max}s\n")
            self.daw_text.insert(tk.END, f"  • License: {license_type}\n")
            self.daw_text.insert(tk.END, f"  • File Type: {filetype}\n")
            self.daw_text.insert(tk.END, f"  • Quality Mode: {'On' if quality_mode else 'Off'}\n")
            self.daw_text.insert(tk.END, f"  • Preference: {'Rating' if prefer_rating else 'Downloads'}\n")
            self.daw_text.insert(tk.END, f"  • Fallback Method: {fallback_method}\n\n")
            self.daw_text.see(tk.END)
        
        # Create progress window with modern styling
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Processing Sounds...")
        
        # Style the progress window
        progress_window.configure(bg=self.MODERN_BG)
        
        # Make window larger and center it
        window_width = 500
        window_height = 200
        
        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Calculate position to center window
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 4)
        
        # Set window geometry with modern styling
        progress_window.geometry(f"{window_width}x{window_height}+{x}+{y}")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        # Prevent closing
        progress_window.protocol("WM_DELETE_WINDOW", lambda: None)
        
        # Center content
        progress_window.grid_columnconfigure(0, weight=1)
        progress_window.grid_rowconfigure(0, weight=1)
        
        # Modern header
        header_label = tk.Label(
            progress_window,
            text="Creating Soundscape",
            bg=self.MODERN_BG,
            fg=self.MODERN_FG,
            font=("Segoe UI", 14, "bold"),
            pady=10
        )
        header_label.pack()
        
        # Subtitle
        subtitle_label = tk.Label(
            progress_window,
            text="Downloading and processing sounds...",
            bg=self.MODERN_BG,
            fg="#aaaaaa",
            font=("Segoe UI", 10),
            pady=5
        )
        subtitle_label.pack()
        
        # Progress bar with modern styling
        progress_container = tk.Frame(progress_window, bg=self.MODERN_BG, padx=30, pady=15)
        progress_container.pack(fill=tk.X)
        
        self.progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(
            progress_container,
            variable=self.progress_var,
            maximum=100,
            length=400,
            mode='determinate'
        )
        progress_bar.pack()
        
        # Status label
        self.progress_status_label = tk.Label(
            progress_window,
            text="Initializing...",
            bg=self.MODERN_BG,
            fg=self.MODERN_FG,
            font=("Segoe UI", 10),
            pady=10
        )
        self.progress_status_label.pack()
        
        # Percentage label
        self.progress_percent_label = tk.Label(
            progress_window,
            text="0%",
            bg=self.MODERN_BG,
            fg=self.MODERN_HIGHLIGHT,
            font=("Segoe UI", 11, "bold"),
            pady=5
        )
        self.progress_percent_label.pack()
        
        # Cancel button (optional - you can add this functionality)
        '''
        cancel_button = tk.Button(
            progress_window,
            text="Cancel",
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_FG,
            font=("Segoe UI", 9),
            relief="flat",
            command=lambda: self._cancel_processing(progress_window)
        )
        cancel_button.pack(pady=10)
        '''
        
        # Create a thread for processing
        def process_sounds_thread():
            try:
                # Call the actual processing function
                self._process_sounds_internal(
                    samplerate, duration_min, duration_max, license_type, 
                    filetype, quality_mode, prefer_rating, fallback_method,
                    self.progress_var, self.progress_status_label, self.progress_percent_label
                )
                
                # Schedule cleanup on main thread
                self.root.after(0, lambda: self._on_sounds_processed_safe(progress_window))
                
            except Exception as e:
                self.root.after(0, lambda error=e: self._on_processing_error(error, progress_window))
        
        # Start processing thread
        processing_thread = threading.Thread(target=process_sounds_thread, daemon=True)
        processing_thread.start()
        
        # Update UI periodically
        self._update_progress_window(progress_window, self.progress_var, 
                                    self.progress_status_label, self.progress_percent_label)

    def _process_sounds_internal(self, samplerate, duration_min, duration_max, 
                           license_type, filetype, quality_mode, 
                           prefer_rating, fallback_method,
                           progress_var, status_label, percent_label):
        """Internal sound processing logic"""
        global processed_tracks, used_sound_ids, downloaded_files
        
        API_KEY = freesound_api_key
        processed_sounds = []
        atmo_sound = None
        atmo_name = None
        
        # Filterbank - FIX: Ensure samplerate is properly converted
        # Convert samplerate to int if it's a string
        if isinstance(samplerate, str):
            if samplerate == "any":
                actual_samplerate = 44100
            elif samplerate.isdigit():
                actual_samplerate = int(samplerate)
            else:
                actual_samplerate = 44100
        else:
            actual_samplerate = samplerate
        
        filterbank = create_bark_filterbank(actual_samplerate)
        
        # Load hierarchy data
        hierarchy_data = []
        if fallback_method == 'CSV':
            csv_path = os.path.join(os.path.dirname(__file__), "tag_hierarchy.csv")
            hierarchy_data = load_tag_hierarchy(csv_path)
            if not hierarchy_data:
                if hasattr(self, 'daw_text') and self.daw_text.winfo_exists():
                    self.daw_text.insert(tk.END, "No tag hierarchy file found\n")
        
        # Update status - searching for atmosphere
        self.root.after(0, lambda: status_label.config(text="Searching for atmosphere sound..."))
        self.root.after(0, lambda: percent_label.config(text="10%"))
        self.root.after(0, lambda: progress_var.set(10))
        
        # Update log
        if hasattr(self, 'daw_text') and self.daw_text.winfo_exists():
            self.daw_text.insert(tk.END, "\nSearching for atmosphere sound...\n")
            self.daw_text.see(tk.END)
        
        # 1. Search for atmosphere sound (scene_and_location)
        if scene_and_location_tags and len(scene_and_location_tags) > 0:
            atmo_tag = scene_and_location_tags[0] if isinstance(scene_and_location_tags, list) else scene_and_location_tags
            
            if hasattr(self, 'daw_text') and self.daw_text.winfo_exists():
                self.daw_text.insert(tk.END, f"  Tag: '{atmo_tag}'\n")
                self.daw_text.see(tk.END)
            
            # Search for atmo sound
            sound = self._search_for_tag(atmo_tag, samplerate, duration_min, duration_max, 
                                    license_type, filetype, quality_mode, prefer_rating,
                                    hierarchy_data, fallback_method, is_atmo=True)
            
            if sound:
                # Process atmo sound
                processed = download_and_process_sound(sound, f"atmo_{atmo_tag}", -1, actual_samplerate,
                                                importance_value=0.0, filterbank=filterbank)
                if processed:
                    used_sound_ids.add(sound['id'])
                    atmo_sound = processed
                    atmo_name = sound['name']
                    
                    if hasattr(self, 'daw_text') and self.daw_text.winfo_exists():
                        self.daw_text.insert(tk.END, f"Found: '{atmo_name}'\n")
                        self.daw_text.insert(tk.END, f"  • Rating: {sound.get('avg_rating', '?'):.1f}\n")
                        self.daw_text.insert(tk.END, f"  • Duration: {sound.get('duration', '?'):.1f}s\n\n")
                        self.daw_text.see(tk.END)
                    
                    processed_sounds.append((atmo_sound, atmo_name, -1))
            else:
                if hasattr(self, 'daw_text') and self.daw_text.winfo_exists():
                    self.daw_text.insert(tk.END, "No atmosphere sound found\n\n")
                    self.daw_text.see(tk.END)
        
        # Update progress
        self.root.after(0, lambda: status_label.config(text="Processing object sounds..."))
        self.root.after(0, lambda: percent_label.config(text="30%"))
        self.root.after(0, lambda: progress_var.set(30))
        
        # 2. Search for object sounds
        all_tags = []
        if isinstance(recognized_tags, list):
            all_tags = recognized_tags
        else:
            # Handle different tag formats
            all_tags = list(recognized_tags) if hasattr(recognized_tags, '__iter__') else []
        
        if hasattr(self, 'daw_text') and self.daw_text.winfo_exists():
            self.daw_text.insert(tk.END, "Processing object sounds:\n")
            self.daw_text.see(tk.END)
        
        for index, original_tag in enumerate(all_tags):
            if index > 0:
                time.sleep(1)  # Rate limiting
            
            progress_percent = 30 + (index / len(all_tags) * 60)
            self.root.after(0, lambda p=progress_percent: progress_var.set(p))
            self.root.after(0, lambda p=progress_percent: percent_label.config(text=f"{int(p)}%"))
            self.root.after(0, lambda t=original_tag: 
                    status_label.config(text=f"Processing '{t}'..."))
            
            if hasattr(self, 'daw_text') and self.daw_text.winfo_exists():
                self.daw_text.insert(tk.END, f"\n[{index+1}/{len(all_tags)}] '{original_tag}'\n")
                self.daw_text.see(tk.END)
            
            # Search for sound
            sound = self._search_for_tag(original_tag, samplerate, duration_min, duration_max,
                                    license_type, filetype, quality_mode, prefer_rating,
                                    hierarchy_data, fallback_method, is_atmo=False)
            
            if not sound:
                if hasattr(self, 'daw_text') and self.daw_text.winfo_exists():
                    self.daw_text.insert(tk.END, f"No sound found\n")
                    self.daw_text.see(tk.END)
                continue
            
            # Get importance value
            object_importance = 0.5
            if 'importance_values' in globals() and importance_values and index < len(importance_values):
                object_importance = importance_values[index]
            elif hasattr(recognized_tags, 'importance') and recognized_tags.importance and index < len(recognized_tags.importance):
                object_importance = recognized_tags.importance[index]
            else:
                object_importance = 0.7 - (index * 0.1)
                object_importance = max(0.3, object_importance)
            
            # Process sound
            processed = download_and_process_sound(sound, original_tag, index+1, actual_samplerate,
                                            importance_value=object_importance, filterbank=filterbank)
            if not processed:
                if hasattr(self, 'daw_text') and self.daw_text.winfo_exists():
                    self.daw_text.insert(tk.END, f"Error processing sound\n")
                    self.daw_text.see(tk.END)
                continue
                
            import os
            sound_dir = os.path.join(BASE_DIR, "downloaded_sounds")
            os.makedirs(sound_dir, exist_ok=True)

            used_sound_ids.add(sound['id'])
            downloaded_files.append(os.path.join(sound_dir, f"sound_{original_tag}_{index+1}.mp3"))
            
            # Apply panning
            pan_value = sound_pannings[index] if index < len(sound_pannings) else 0.0
            panned = processed.pan(pan_value)
            
            processed_sounds.append((panned, sound['name'], index))
            
            if hasattr(self, 'daw_text') and self.daw_text.winfo_exists():
                self.daw_text.insert(tk.END, f"Found: '{sound['name']}'\n")
                self.daw_text.insert(tk.END, f"  • Rating: {sound.get('avg_rating', '?'):.1f}\n")
                self.daw_text.insert(tk.END, f"  • Downloads: {sound.get('num_downloads', '?'):.1f}\n")
                self.daw_text.insert(tk.END, f"  • Duration: {sound.get('duration', '?'):.1f}s\n")
                self.daw_text.insert(tk.END, f"  • Pan: {pan_value:.2f}\n")
                self.daw_text.insert(tk.END, f"  • Importance: {object_importance:.2f}\n")
                self.daw_text.see(tk.END)
        
        # 3. Create timeline and tracks
        self.root.after(0, lambda: status_label.config(text="Creating timeline and mix..."))
        self.root.after(0, lambda: percent_label.config(text="95%"))
        self.root.after(0, lambda: progress_var.set(95))
        
        if not processed_sounds:
            if hasattr(self, 'daw_text') and self.daw_text.winfo_exists():
                self.daw_text.insert(tk.END, "\nNo sounds were processed successfully\n")
                self.daw_text.see(tk.END)
            return
        
        # Determine base duration
        if atmo_sound:
            base_duration = len(atmo_sound)
            base_sound = atmo_sound
            base_name = atmo_name
        else:
            # Find longest sound
            longest_sound, longest_name, _ = max(processed_sounds, key=lambda x: len(x[0]))
            base_duration = len(longest_sound)
            base_sound = longest_sound
            base_name = longest_name
        
        if hasattr(self, 'daw_text') and self.daw_text.winfo_exists():
            self.daw_text.insert(tk.END, f"\nCreating Timeline\n")
            self.daw_text.insert(tk.END, f"Base track: '{base_name}' ({base_duration/1000:.1f}s)\n\n")
            self.daw_text.see(tk.END)
        
        # Create tracks with random positioning
        for sound, name, sound_idx in processed_sounds:
            if sound_idx == -1:  # Atmo sound
                position = 0
                pan = 0.0
                is_atmo = True
            else:  # Object sound
                max_position = base_duration - len(sound)
                if max_position > 0:
                    position = random.randint(0, max_position)
                else:
                    position = 0
            
                pan = sound_pannings[sound_idx] if sound_idx < len(sound_pannings) else 0.0
                is_atmo = False
            
                # Display timeline info
                start_sec = position / 1000
                end_sec = (position + len(sound)) / 1000
                if hasattr(self, 'daw_text') and self.daw_text.winfo_exists():
                    self.daw_text.insert(tk.END, f"  • {name}: {start_sec:.1f}s-{end_sec:.1f}s ")
                    self.daw_text.insert(tk.END, f"(pan: {pan:.1f})\n")
                    self.daw_text.see(tk.END)
            
            # Add to processed tracks
            processed_tracks.append({
                'audio': sound,
                'name': name,
                'index': sound_idx,
                'position': position,
                'pan': pan,
                'is_atmo': is_atmo
            })
        
        # Run masking analysis
        if hasattr(self, 'daw_text') and self.daw_text.winfo_exists():
            self.daw_text.insert(tk.END, "\n🎛️ Running masking analysis...\n")
            self.daw_text.see(tk.END)
        
        # Find overlaps and apply EQ
        overlap_pairs = []
        
        # Check for overlaps between sounds
        for i, track1 in enumerate(processed_tracks):
            for j, track2 in enumerate(processed_tracks[i+1:], i+1):
                if not track1['is_atmo'] and not track2['is_atmo']:  # Only object sounds
                    start1 = track1['position']
                    end1 = start1 + len(track1['audio'])
                    start2 = track2['position']
                    end2 = start2 + len(track2['audio'])
                    
                    # Check if sounds overlap
                    if not (end1 <= start2 or end2 <= start1):
                        overlap_start = max(start1, start2)
                        overlap_end = min(end1, end2)
                        overlap_pairs.append((
                            track1, track2,
                            overlap_start, overlap_end
                        ))
        
        # Apply EQ to overlapping pairs
        if overlap_pairs:
            if hasattr(self, 'daw_text') and self.daw_text.winfo_exists():
                self.daw_text.insert(tk.END, f"Found {len(overlap_pairs)} overlapping sound pairs\n")
            
            for track1, track2, start_ms, end_ms in overlap_pairs:
                if hasattr(self, 'daw_text') and self.daw_text.winfo_exists():
                    self.daw_text.insert(tk.END, f"  Overlap: {track1['name']} & {track2['name']} ")
                    self.daw_text.insert(tk.END, f"({start_ms/1000:.2f}s-{end_ms/1000:.2f}s)\n")
                
                # Get overlapping segments
                seg1_start = start_ms - track1['position']
                seg1_end = end_ms - track1['position']
                seg2_start = start_ms - track2['position']
                seg2_end = end_ms - track2['position']
                
                seg1 = track1['audio'][seg1_start:seg1_end]
                seg2 = track2['audio'][seg2_start:seg2_end]
                
                # Analyze energies
                energy1 = analyze_bark_energy(seg1, filterbank, actual_samplerate)
                energy2 = analyze_bark_energy(seg2, filterbank, actual_samplerate)
                
                # Design EQ for masking reduction
                eq_settings = design_eq_for_masking(energy1, energy2, filterbank)
                
                if eq_settings:
                    # Apply EQ to the quieter sound (track2 in this case)
                    before = track2['audio'][:seg2_start]
                    after = track2['audio'][seg2_end:]
                    eq_segment = apply_eq_to_segment(track2['audio'][seg2_start:seg2_end], 
                                                    eq_settings, actual_samplerate)
                    
                    # Update the track with EQ applied
                    track2['audio'] = before + eq_segment + after
                    if hasattr(self, 'daw_text') and self.daw_text.winfo_exists():
                        self.daw_text.insert(tk.END, f"    Applied EQ to {track2['name']}\n")
        else:
            if hasattr(self, 'daw_text') and self.daw_text.winfo_exists():
                self.daw_text.insert(tk.END, "No overlapping sounds found\n")
        
        if hasattr(self, 'daw_text') and self.daw_text.winfo_exists():
            self.daw_text.see(tk.END)
        
        self.root.after(0, lambda: percent_label.config(text="100%"))
        self.root.after(0, lambda: progress_var.set(100))

    def _search_for_tag(self, original_tag, samplerate, duration_min, duration_max,
                   license_type, filetype, quality_mode, prefer_rating,
                   hierarchy_data, fallback_method, is_atmo=False):
        """Search for a sound matching the tag"""
        API_KEY = freesound_api_key
        
        sound = None
        current_tag = original_tag
        tried_tags = set()
        openai_fallback_tries = 0
        
        while current_tag and current_tag not in tried_tags:
            tried_tags.add(current_tag)
            
            # Define search tiers based on quality mode
            rating_steps = [5.0, 4.0, 3.5, 3.0, 2.5, 2.0, 0] if quality_mode and prefer_rating else [0]
            download_steps = [1000, 500, 100, 50, 0] if quality_mode and not prefer_rating else [0]
            
            found_valid_sound = False
            
            # Only check first page (10 results) - remove the round loop
            for rating in rating_steps:
                if found_valid_sound:
                    break
                for downloads in download_steps:
                    if found_valid_sound:
                        break
                    
                    # Search for sound - only page 1
                    results = search_sounds(
                        API_KEY, current_tag, samplerate,
                        duration_min, duration_max, license_type, filetype, 
                        rating, page=1  # FIX: Only check page 1
                    )
                    
                    if results:
                        # Find a sound not already used
                        for candidate_sound in results:
                            if candidate_sound['id'] in used_sound_ids:
                                continue
                            
                            # Description check
                            if DESCRIPTION_CHECK_ENABLED and 'description' in candidate_sound:
                                description = candidate_sound['description']
                                if description and len(description.strip()) > 0:
                                    img_desc = image_input_description[0] if image_input_description else "No description"
                                    is_valid = check_description_with_openai(description, current_tag, img_desc)
                                    if not is_valid:
                                        continue
                            
                            # Found valid sound
                            sound = candidate_sound
                            found_valid_sound = True
                            break
                    
                    if found_valid_sound:
                        break
            
            if found_valid_sound:
                break
            else:
                # Try fallback
                if fallback_method == 'CSV':
                    current_tag = get_fallback_tag(current_tag, hierarchy_data)
                    if current_tag:
                        print(f"Trying fallback: '{current_tag}'")
                else:
                    openai_fallback_tries += 1
                    if openai_fallback_tries >= MAX_OPENAI_FALLBACK_TRIES:
                        current_tag = None
                    else:
                        current_tag = get_openai_fallback_tag(current_tag)
                        if current_tag:
                            print(f"OpenAI fallback: '{current_tag}'")
        
        return sound

    def _on_processing_error(self, error, progress_window):
        """Callback when processing fails"""
        # Add error message to DAW text BEFORE destroying window
        self.daw_text.insert(tk.END, f"\n✗ Error during processing: {error}\n")
        self.daw_text.see(tk.END)
        self.update_status("Error in sound processing")
        
        # Now destroy the progress window
        progress_window.destroy()

    def _update_progress_window(self, progress_window, progress_var, status_label, percent_label):
        """Update progress window"""
        if progress_window.winfo_exists():
            progress_window.after(100, lambda: self._update_progress_window(
                progress_window, progress_var, status_label, percent_label))
        
    def setup_tab5(self):
        """Setup Feedback tab"""
        self.tab5 = tk.Frame(self.notebook, bg=self.MODERN_BG)
        self.notebook.add(self.tab5, text="Feedback")
        
        # Main container with padding
        main_container = tk.Frame(self.tab5, bg=self.MODERN_BG)
        main_container.pack(fill=tk.BOTH, expand=True, padx=30, pady=25)
        
        # Feedback header
        header_frame = tk.Frame(main_container, bg=self.MODERN_BG)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        header_icon = tk.Label(
            header_frame,
            text="💬",
            bg=self.MODERN_BG,
            fg=self.MODERN_ACCENT,
            font=("Segoe UI", 28)
        )
        header_icon.pack(side=tk.LEFT, padx=(0, 15))
        
        header_text = tk.Label(
            header_frame,
            text="Share Your Experience",
            bg=self.MODERN_BG,
            fg=self.MODERN_FG,
            font=("Segoe UI", 18, "bold")
        )
        header_text.pack(side=tk.LEFT)
        
        # Feedback container - modern version
        feedback_frame = tk.LabelFrame(
            main_container,
            text="Feedback & Rating",
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_FG,
            font=("Segoe UI", 13, "bold"),
            relief="flat",
            bd=1,
            padx=25,
            pady=25
        )
        feedback_frame.pack(fill=tk.BOTH, expand=True)
        
        # Info label with modern styling
        info_text = """We value your feedback! Please rate your experience and share any thoughts about the sound creation process. Your input helps us improve."""
        
        info_label = tk.Label(
            feedback_frame,
            text=info_text,
            bg=self.MODERN_SECONDARY,
            fg="#aaaaaa",
            font=("Segoe UI", 11),
            wraplength=600,
            justify=tk.LEFT,
            pady=15
        )
        info_label.pack(fill=tk.X, pady=(0, 20))
        
        # Rating section
        rating_container = tk.Frame(feedback_frame, bg=self.MODERN_SECONDARY)
        rating_container.pack(fill=tk.X, pady=(0, 25))
        
        # Rating label
        rating_label = tk.Label(
            rating_container,
            text="Rate your experience:",
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_FG,
            font=("Segoe UI", 12, "bold"),
            anchor=tk.W
        )
        rating_label.pack(fill=tk.X, pady=(0, 15))
        
        # Rating stars container
        self.rating_frame = tk.Frame(rating_container, bg=self.MODERN_SECONDARY)
        self.rating_frame.pack()
        
        self.rating_vars = []
        self.rating_buttons = []
        
        # Modern star rating buttons
        for i in range(1, 6):
            var = tk.StringVar(value="☆")
            btn = tk.Button(
                self.rating_frame,
                textvariable=var,
                width=3,
                bg=self.MODERN_SECONDARY,
                fg=self.MODERN_ACCENT,
                font=("Segoe UI", 22),
                relief="flat",
                borderwidth=0,
                padx=2,
                pady=0,
                cursor="hand2",
                activebackground=self.MODERN_SECONDARY,
                activeforeground=self.MODERN_HIGHLIGHT,
                command=lambda idx=i: self.set_rating(idx)
            )
            btn.pack(side=tk.LEFT, padx=4)
            self.rating_vars.append(var)
            self.rating_buttons.append(btn)
        
        # Rating labels below stars
        labels_frame = tk.Frame(rating_container, bg=self.MODERN_SECONDARY)
        labels_frame.pack(pady=10)
        
        left_label = tk.Label(
            labels_frame,
            text="Poor",
            bg=self.MODERN_SECONDARY,
            fg="#888888",
            font=("Segoe UI", 9)
        )
        left_label.pack(side=tk.LEFT, padx=(0, 120))
        
        right_label = tk.Label(
            labels_frame,
            text="Excellent",
            bg=self.MODERN_SECONDARY,
            fg="#888888",
            font=("Segoe UI", 9)
        )
        right_label.pack(side=tk.RIGHT, padx=(120, 0))
        
        # Selected rating display
        self.rating_display = tk.Label(
            rating_container,
            text="Select a rating",
            bg=self.MODERN_SECONDARY,
            fg="#888888",
            font=("Segoe UI", 10),
            pady=5
        )
        self.rating_display.pack()
        
        # Feedback text section
        feedback_text_container = tk.Frame(feedback_frame, bg=self.MODERN_SECONDARY)
        feedback_text_container.pack(fill=tk.BOTH, expand=True, pady=(0, 25))
        
        # Feedback label
        feedback_label = tk.Label(
            feedback_text_container,
            text="Your feedback (optional):",
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_FG,
            font=("Segoe UI", 12, "bold"),
            anchor=tk.W
        )
        feedback_label.pack(fill=tk.X, pady=(0, 10))
        
        # Modern text area with scrollbar
        text_frame = tk.Frame(feedback_text_container, bg=self.MODERN_SECONDARY)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.feedback_text = tk.Text(
            text_frame,
            height=6,
            bg=self.MODERN_BG,
            fg=self.MODERN_FG,
            insertbackground=self.MODERN_FG,
            font=("Segoe UI", 10),
            relief="flat",
            wrap=tk.WORD,
            padx=15,
            pady=15
        )
        
        # Add scrollbar
        text_scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.feedback_text.yview)
        self.feedback_text.configure(yscrollcommand=text_scrollbar.set)
        
        self.feedback_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Placeholder text
        placeholder_text = "Share your thoughts about the sound creation experience, suggestions for improvement, or anything else you'd like to tell us..."
        
        self.feedback_text.insert("1.0", placeholder_text)
        self.feedback_text.config(fg="#888888")
        
        # Bind events for placeholder behavior
        def on_focus_in(event):
            if self.feedback_text.get("1.0", "end-1c") == placeholder_text:
                self.feedback_text.delete("1.0", tk.END)
                self.feedback_text.config(fg=self.MODERN_FG)
        
        def on_focus_out(event):
            if not self.feedback_text.get("1.0", "end-1c").strip():
                self.feedback_text.insert("1.0", placeholder_text)
                self.feedback_text.config(fg="#888888")
        
        self.feedback_text.bind("<FocusIn>", on_focus_in)
        self.feedback_text.bind("<FocusOut>", on_focus_out)
        
        # Send button container
        button_container = tk.Frame(feedback_frame, bg=self.MODERN_SECONDARY)
        button_container.pack(pady=10)
        
        # Modern send button
        send_button = tk.Button(
            button_container,
            text="📤 Submit Feedback",
            command=self.send_feedback,
            bg=self.MODERN_ACCENT,
            fg=self.MODERN_FG,
            font=("Segoe UI", 12, "bold"),
            relief="flat",
            borderwidth=0,
            padx=35,
            pady=14,
            cursor="hand2",
            activebackground="#3a8eff",
            activeforeground=self.MODERN_FG
        )
        send_button.pack()
        
        # Status label for feedback submission
        self.feedback_status_label = tk.Label(
            button_container,
            text="",
            bg=self.MODERN_SECONDARY,
            fg="#888888",
            font=("Segoe UI", 9),
            pady=5
        )
        self.feedback_status_label.pack()
        
        # Privacy note
        privacy_frame = tk.Frame(feedback_frame, bg=self.MODERN_SECONDARY, pady=15)
        privacy_frame.pack(fill=tk.X)
        
        privacy_label = tk.Label(
            privacy_frame,
            text="🔒 Your feedback is anonymous and will be used solely to improve the application.",
            bg=self.MODERN_SECONDARY,
            fg="#666666",
            font=("Segoe UI", 9),
            justify=tk.CENTER
        )
        privacy_label.pack()
        
    def set_rating(self, rating):
        """Set the rating stars with modern feedback"""
        global current_rating
        current_rating = rating
        
        # Update stars
        for i in range(5):
            if i < rating:
                self.rating_vars[i].set("★")
                # Change color based on rating
                if rating >= 4:
                    self.rating_buttons[i].config(fg=self.MODERN_HIGHLIGHT)  # Green for high ratings
                elif rating >= 3:
                    self.rating_buttons[i].config(fg=self.MODERN_ACCENT)     # Blue for medium ratings
                else:
                    self.rating_buttons[i].config(fg=self.MODERN_DANGER)     # Red for low ratings
            else:
                self.rating_vars[i].set("☆")
                self.rating_buttons[i].config(fg="#666666")  # Gray for unselected
        
        # Update rating display text
        rating_texts = {
            1: "Poor - Needs significant improvement",
            2: "Fair - Has room for improvement", 
            3: "Good - Satisfactory experience",
            4: "Very Good - Enjoyable experience",
            5: "Excellent - Outstanding experience"
        }
        
        if hasattr(self, 'rating_display'):
            self.rating_display.config(
                text=rating_texts.get(rating, "Select a rating"),
                fg=self.MODERN_HIGHLIGHT if rating >= 4 else 
                self.MODERN_ACCENT if rating >= 3 else 
                self.MODERN_DANGER
            )
                
    def send_feedback(self):
        """Send feedback via email with modern UI"""
        global current_rating
        
        if current_rating == 0:
            messagebox.showwarning("No Rating", "Please select a rating first!")
            return
            
        feedback = self.feedback_text.get("1.0", tk.END).strip()
        
        # Remove placeholder if it's still there
        if feedback == "Share your thoughts about the sound creation experience, suggestions for improvement, or anything else you'd like to tell us...":
            feedback = ""
        
        # Show sending message with modern styling
        sending_window = tk.Toplevel(self.root)
        sending_window.title("Sending Feedback...")
        sending_window.configure(bg=self.MODERN_BG)
        
        # Center the window
        window_width = 350
        window_height = 150
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        sending_window.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        sending_window.transient(self.root)
        sending_window.grab_set()
        
        # Modern sending indicator
        sending_icon = tk.Label(
            sending_window,
            text="📤",
            bg=self.MODERN_BG,
            fg=self.MODERN_ACCENT,
            font=("Segoe UI", 24)
        )
        sending_icon.pack(pady=10)
        
        sending_label = tk.Label(
            sending_window,
            text="Sending your feedback...",
            bg=self.MODERN_BG,
            fg=self.MODERN_FG,
            font=("Segoe UI", 11)
        )
        sending_label.pack(pady=5)
        
        # Progress indicator
        progress_bar = ttk.Progressbar(
            sending_window,
            mode='indeterminate',
            length=200
        )
        progress_bar.pack(pady=15)
        progress_bar.start()
        
        sending_window.update()
        
        # Send feedback in a separate thread to keep UI responsive
        def send_feedback_thread():
            try:
                success = send_feedback_email(current_rating, feedback)
                
                # Update UI on main thread
                self.root.after(0, lambda: self._on_feedback_sent(success, sending_window))
                
            except Exception as e:
                self.root.after(0, lambda error=e: self._on_feedback_error(error, sending_window))
        
        # Start sending thread
        feedback_thread = threading.Thread(target=send_feedback_thread, daemon=True)
        feedback_thread.start()

    def _on_feedback_sent(self, success, sending_window):
        """Callback when feedback is sent successfully"""
        # Stop progress bar
        for widget in sending_window.winfo_children():
            if isinstance(widget, ttk.Progressbar):
                widget.stop()
        
        # Update sending window
        if success:
            for widget in sending_window.winfo_children():
                if isinstance(widget, tk.Label) and "Sending your feedback" in widget.cget("text"):
                    widget.config(text="✓ Feedback sent successfully!", fg=self.MODERN_SUCCESS)
                    break
            
            # Update status label in main tab
            if hasattr(self, 'feedback_status_label'):
                self.feedback_status_label.config(
                    text="✓ Thank you for your feedback!",
                    fg=self.MODERN_SUCCESS
                )
            
            # Clear after 3 seconds
            self.root.after(3000, lambda: 
                self.feedback_status_label.config(text="", fg="#888888") 
                if hasattr(self, 'feedback_status_label') else None
            )
            
            # Reset form
            self.root.after(1000, lambda: self._reset_feedback_form(sending_window))
            
        else:
            self._on_feedback_error("Failed to send feedback", sending_window)

    def _on_feedback_error(self, error, sending_window):
        """Callback when feedback sending fails"""
        # Stop progress bar
        for widget in sending_window.winfo_children():
            if isinstance(widget, ttk.Progressbar):
                widget.stop()
        
        # Update sending window
        for widget in sending_window.winfo_children():
            if isinstance(widget, tk.Label) and ("Sending your feedback" in widget.cget("text") or "✓ Feedback sent" in widget.cget("text")):
                widget.config(text="✗ Failed to send feedback", fg=self.MODERN_DANGER)
                break
        
        # Update status label in main tab
        if hasattr(self, 'feedback_status_label'):
            self.feedback_status_label.config(
                text="✗ Failed to send feedback",
                fg=self.MODERN_DANGER
            )
        
        # Close window after delay
        self.root.after(2000, sending_window.destroy)

    def _reset_feedback_form(self, sending_window):
        """Reset the feedback form after successful submission"""
        # Close sending window
        sending_window.destroy()
        
        # Reset form
        self.set_rating(0)
        self.feedback_text.delete("1.0", tk.END)
        
        # Restore placeholder
        placeholder_text = "Share your thoughts about the sound creation experience, suggestions for improvement, or anything else you'd like to tell us..."
        self.feedback_text.insert("1.0", placeholder_text)
        self.feedback_text.config(fg="#888888")
        
        self.update_status("Feedback submitted successfully")

    def _create_daw_controls(self):
        """Create modern DAW control interface"""
        # Clear existing controls from container
        for widget in self.daw_controls_container.winfo_children():
            widget.destroy()
        
        if not processed_tracks:
            # Modern placeholder
            placeholder_frame = tk.Frame(self.daw_controls_container, bg=self.MODERN_SECONDARY)
            placeholder_frame.pack(fill=tk.BOTH, expand=True, pady=40)
            
            placeholder_label = tk.Label(
                placeholder_frame,
                text="No tracks available\n\nProcess an image to create soundscape",
                bg=self.MODERN_SECONDARY,
                fg="#666666",
                font=("Segoe UI", 11),
                justify=tk.CENTER
            )
            placeholder_label.pack()
            self._update_daw_status("No tracks available")
            
            # Clear reverb buttons if room was previously detected
            for widget in self.reverb_container.winfo_children():
                widget.destroy()
            return
        
        # Main container with modern styling
        main_daw_frame = tk.Frame(
            self.daw_controls_container,
            bg=self.MODERN_BG
        )
        main_daw_frame.pack(fill=tk.BOTH, expand=True)
        
        # UPPER PART: Track sliders with modern styling
        tracks_frame = tk.LabelFrame(
            main_daw_frame,
            text="Track Controls",
            bg=self.MODERN_SECONDARY,
            fg=self.MODERN_FG,
            font=("Segoe UI", 11, "bold"),
            relief="flat",
            bd=1,
            padx=15,
            pady=10
        )
        tracks_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Canvas with scrollbar for many tracks - modern styling
        canvas = tk.Canvas(
            tracks_frame,
            bg=self.MODERN_SECONDARY,
            highlightthickness=0
        )
        
        scrollbar = ttk.Scrollbar(
            tracks_frame,
            orient=tk.VERTICAL,
            command=canvas.yview,
            style="Vertical.TScrollbar"
        )
        
        scrollable_frame = tk.Frame(canvas, bg=self.MODERN_SECONDARY)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Store volume sliders
        self.volume_sliders = []
        
        # Create modern controls for each track
        for i, track in enumerate(processed_tracks):
            # Track container with subtle modern border
            track_frame = tk.Frame(
                scrollable_frame,
                bg=self.MODERN_SECONDARY,
                highlightbackground="#333333",
                highlightthickness=1,
                padx=10,
                pady=8
            )
            track_frame.pack(fill=tk.X, pady=3, padx=5)
            
            # Track header with type icon and name
            header_row = tk.Frame(track_frame, bg=self.MODERN_SECONDARY)
            header_row.pack(fill=tk.X, pady=(0, 8))
            
            # Track type with color coding
            if track['is_atmo']:
                track_type = "Atmo"
                type_color = self.MODERN_HIGHLIGHT
                type_text = "Atmosphere"
            else:
                track_type = "Audio"
                type_color = self.MODERN_ACCENT
                type_text = "Object"
            
            type_label = tk.Label(
                header_row,
                text=f"{track_type} {type_text}",
                bg=self.MODERN_SECONDARY,
                fg=type_color,
                font=("Segoe UI", 10, "bold"),
                width=15,
                anchor=tk.W
            )
            type_label.pack(side=tk.LEFT, padx=(0, 10))
            
            # Track name with ellipsis for long names
            display_name = track['name'][:35] + "..." if len(track['name']) > 35 else track['name']
            name_label = tk.Label(
                header_row,
                text=display_name,
                bg=self.MODERN_SECONDARY,
                fg=self.MODERN_FG,
                font=("Segoe UI", 10),
                anchor=tk.W
            )
            name_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            # Controls row
            controls_row = tk.Frame(track_frame, bg=self.MODERN_SECONDARY)
            controls_row.pack(fill=tk.X)
            
            # Volume control with modern styling
            volume_container = tk.Frame(controls_row, bg=self.MODERN_SECONDARY)
            volume_container.pack(side=tk.LEFT, padx=(0, 15))
            
            volume_label = tk.Label(
                volume_container,
                text="Volume:",
                bg=self.MODERN_SECONDARY,
                fg="#aaaaaa",
                font=("Segoe UI", 9),
                width=8,
                anchor=tk.W
            )
            volume_label.pack(side=tk.LEFT, padx=(0, 5))
            
            volume_var = tk.DoubleVar(value=1.0)
            
            # Modern volume slider
            volume_slider = ttk.Scale(
                volume_container,
                from_=0.0,
                to=2.0,
                variable=volume_var,
                orient=tk.HORIZONTAL,
                length=100
            )
            volume_slider.pack(side=tk.LEFT, padx=(0, 8))
            
            # Volume value display
            volume_value = tk.Label(
                volume_container,
                text="1.00",
                bg=self.MODERN_SECONDARY,
                fg=self.MODERN_HIGHLIGHT,
                font=("Segoe UI", 9, "bold"),
                width=5
            )
            volume_value.pack(side=tk.LEFT)
            
            # Update label when slider changes
            def update_volume_label(var, label):
                return lambda *args: label.config(text=f"{var.get():.2f}")
            
            volume_var.trace_add("write", update_volume_label(volume_var, volume_value))
            
            # Pan indicator (only for object tracks)
            if not track['is_atmo']:
                pan_container = tk.Frame(controls_row, bg=self.MODERN_SECONDARY)
                pan_container.pack(side=tk.LEFT, padx=(0, 15))
                
                pan_label = tk.Label(
                    pan_container,
                    text=f"Pan: {track['pan']:.2f}",
                    bg=self.MODERN_SECONDARY,
                    fg="#aaaaaa",
                    font=("Segoe UI", 9),
                    width=10,
                    anchor=tk.W
                )
                pan_label.pack(side=tk.LEFT)
            
            # Action buttons container
            actions_container = tk.Frame(controls_row, bg=self.MODERN_SECONDARY)
            actions_container.pack(side=tk.RIGHT)
            
            # Replace sound button - modern styling
            replace_btn = tk.Button(
                actions_container,
                text="🔄 Replace",
                command=lambda idx=i: self._replace_and_refresh(idx),
                bg=self.MODERN_SECONDARY,
                fg=self.MODERN_FG,
                font=("Segoe UI", 9),
                relief="flat",
                borderwidth=0,
                padx=10,
                pady=3,
                cursor="hand2",
                activebackground="#333333",
                activeforeground=self.MODERN_FG
            )
            replace_btn.pack(side=tk.LEFT, padx=2)
            
            # Mute button - modern styling
            mute_btn = tk.Button(
                actions_container,
                text="🔇 Mute",
                command=lambda var=volume_var: var.set(0.0),
                bg=self.MODERN_SECONDARY,
                fg=self.MODERN_FG,
                font=("Segoe UI", 9),
                relief="flat",
                borderwidth=0,
                padx=10,
                pady=3,
                cursor="hand2",
                activebackground="#333333",
                activeforeground=self.MODERN_FG
            )
            mute_btn.pack(side=tk.LEFT, padx=2)
            
            # Store slider reference
            self.volume_sliders.append({
                'index': i,
                'var': volume_var,
                'slider': volume_slider,
                'track': track
            })
        
        # Pack canvas and scrollbar
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configure mouse wheel for track scrolling
        def _on_track_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_track_mousewheel)
        
        # Reverb buttons container (if room detected) - FIXED: Update existing container
        for widget in self.reverb_container.winfo_children():
            widget.destroy()
        
        if self._check_room_detection():
            # Add separator
            separator = ttk.Separator(self.reverb_container, orient=tk.HORIZONTAL)
            separator.pack(fill=tk.X, pady=5)
            
            reverb_frame = tk.Frame(self.reverb_container, bg=self.MODERN_SECONDARY)
            reverb_frame.pack(pady=5)
            
            # Reverb label
            reverb_label = tk.Label(
                reverb_frame,
                text="Reverb",
                bg=self.MODERN_BG,
                fg=self.MODERN_FG,
                font=("Segoe UI", 11, "bold"),
                padx=10   # ← safest
            )

            reverb_label.pack(side=tk.LEFT)
            
            # Add reverb button
            add_reverb_btn = tk.Button(
                reverb_frame,
                text="🎚️ Add",
                command=self._apply_reverb,
                bg=self.MODERN_HIGHLIGHT,
                fg=self.MODERN_BG,
                font=("Segoe UI", 9),
                relief="flat",
                borderwidth=0,
                padx=12,
                pady=5,
                cursor="hand2",
                activebackground="#00c49a",
                activeforeground=self.MODERN_BG
            )
            add_reverb_btn.pack(side=tk.LEFT, padx=5)
            
            # Remove reverb button
            remove_reverb_btn = tk.Button(
                reverb_frame,
                text="🔇 Remove",
                command=self._remove_reverb,
                bg=self.MODERN_SECONDARY,
                fg=self.MODERN_FG,
                font=("Segoe UI", 9),
                relief="flat",
                borderwidth=0,
                padx=12,
                pady=5,
                cursor="hand2",
                activebackground="#333333",
                activeforeground=self.MODERN_FG
            )
            remove_reverb_btn.pack(side=tk.LEFT, padx=5)
            
            # Show room parameters if available
            room_params = self._get_room_parameters()
            if room_params:
                room_info = tk.Label(
                    reverb_frame,
                    text=f"🏠 {' | '.join(room_params)}",
                    bg=self.MODERN_SECONDARY,
                    fg="#888888",
                    font=("Segoe UI", 9),
                    padx=15
                )
                room_info.pack(side=tk.LEFT)
        
        # Update status
        track_count = len(processed_tracks)
        atmo_count = sum(1 for track in processed_tracks if track['is_atmo'])
        object_count = track_count - atmo_count
        
        self._update_daw_status(f"✓ {track_count} tracks ({atmo_count} atmosphere, {object_count} objects)")

    def _create_current_mix(self):
        """Create current mix with volume adjustments"""
        if not processed_tracks or not hasattr(self, 'volume_sliders'):
            return None
        
        # Find longest track
        max_duration = max(len(track['audio']) for track in processed_tracks)
        
        # Create silent base track
        try:
            base_track = AudioSegment.silent(duration=max_duration)
        except:
            # Fallback
            base_track = AudioSegment.silent(duration=10000)
        
        # Overlay all tracks with volume adjustments
        for slider_data in self.volume_sliders:
            track_index = slider_data['index']
            volume_factor = slider_data['var'].get()
            
            if track_index >= len(processed_tracks):
                continue
            
            track_data = processed_tracks[track_index]
            audio = track_data['audio']
            
            # Apply volume
            if volume_factor != 1.0:
                gain_change_db = 20 * np.log10(volume_factor)
                audio = audio.apply_gain(gain_change_db)
            
            # Apply panning (if not atmo)
            if not track_data['is_atmo'] and track_data['pan'] != 0.0:
                audio = audio.pan(track_data['pan'])
            
            # Add to mix
            position = track_data.get('position', 0)
            base_track = base_track.overlay(audio, position=position)
        
        return base_track
    
    def _update_playback_status(self, status):
        """Update playback status in UI"""
        if hasattr(self, 'daw_text') and self.daw_text.winfo_exists():
            # Remove previous "Playing..." if exists
            current_text = self.daw_text.get("1.0", tk.END)
            if "Playing..." in current_text and status != "Playing...":
                # Don't add duplicate status
                return
            
            if status not in current_text:
                self.daw_text.insert(tk.END, f"   {status}\n")
                self.daw_text.see(tk.END)
        
        # Also update DAW status label if it exists
        if hasattr(self, 'daw_status_label') and self.daw_status_label.winfo_exists():
            self.daw_status_label.config(text=status)

    def _start_playback_status_updates(self):
        """Start checking playback status"""
        global is_playing
        
        if is_playing:
            # Check if sounddevice is still playing
            try:
                if sd.get_stream() and sd.get_stream().active:
                    # Still playing, check again in 100ms
                    self.root.after(100, self._start_playback_status_updates)
                else:
                    # Playback finished
                    is_playing = False
                    self._update_playback_status("Ready")
            except:
                # No stream active
                is_playing = False
                self._update_playback_status("Ready")

    def _play_mix(self):
        """Play the current mix using sounddevice"""
        global is_playing, reverb_enabled, current_mix_with_reverb
        
        if is_playing:
            self._stop_mix()
            return
        
        current_mix = self._create_current_mix()
        if not current_mix:
            messagebox.showwarning("No Mix", "No mix available to play")
            return
        
        is_playing = True

        def play_audio():
            try:
                # Check if reverb is enabled
                mix_to_play = current_mix
                if reverb_enabled and current_mix_with_reverb:
                    mix_to_play = current_mix_with_reverb
                    print("Playing mix WITH reverb")
                else:
                    print("Playing mix WITHOUT reverb")
                
                # Save to temporary file
                import tempfile
                import os
                
                # Create temp directory if it doesn't exist
                temp_dir = os.path.join(BASE_DIR, "temp_playback")
                os.makedirs(temp_dir, exist_ok=True)
                
                # Create temporary file
                temp_file = os.path.join(temp_dir, "current_mix.wav")
                
                # Export the mix to WAV - FIXED: Ensure we have audio data
                # Convert AudioSegment to array first to check
                samples = np.array(mix_to_play.get_array_of_samples(), dtype=np.float32)
                if len(samples) == 0:
                    print("ERROR: No samples in mix!")
                    self.root.after(0, lambda: self.daw_text.insert(tk.END, "ERROR: Mix has no audio data!\n"))
                    self.root.after(0, lambda: self.daw_text.see(tk.END))
                    is_playing = False
                    return
                
                # Check duration
                duration_sec = len(mix_to_play) / 1000.0
                print(f"Mix duration: {duration_sec:.2f} seconds, {len(samples)} samples")
                
                # Export
                mix_to_play.export(temp_file, format="wav")
                
                # Load with soundfile
                data, samplerate = sf.read(temp_file)
                
                if len(data) == 0:
                    print("ERROR: Loaded audio file is empty!")
                    is_playing = False
                    return
                
                # Normalize volume if needed
                max_val = np.max(np.abs(data))
                print(f"Max amplitude: {max_val:.3f}")
                
                if max_val > 0:
                    # Apply some gain if too quiet
                    if max_val < 0.1:
                        gain = 0.5 / max_val
                        data = data * gain
                    # Reduce if too loud
                    elif max_val > 0.95:
                        data = data * 0.95 / max_val
                
                # Convert to stereo if mono
                if len(data.shape) == 1:
                    data = np.column_stack((data, data))
                
                # Play the audio with blocking=False so we can update status
                print(f"Playing audio: {data.shape[0]} samples, {samplerate}Hz, {data.shape[1]} channels")
                
                # Update status in main thread
                self.root.after(0, lambda: self._update_playback_status(f"Playing ({duration_sec:.1f}s)..."))
                
                # Play with sounddevice
                sd.play(data, samplerate, blocking=True)  # Blocking until done
                
                # Clean up
                try:
                    os.remove(temp_file)
                except:
                    pass
                
                # Reset playing status
                is_playing = False
                
                # Update status
                self.root.after(0, lambda: self._update_playback_status("Finished"))
                
            except Exception as e:
                print(f"Error playing audio: {e}")
                import traceback
                traceback.print_exc()
                is_playing = False
                self.root.after(0, lambda: self._update_playback_status(f"Error: {str(e)[:50]}"))
        
        # Start playback in separate thread
        audio_thread = threading.Thread(target=play_audio, daemon=True, name="PlaybackThread")
        audio_thread.start()
        
        if reverb_enabled:
            self.daw_text.insert(tk.END, "▶ Playing mix WITH REVERB...\n")
        else:
            self.daw_text.insert(tk.END, "▶ Playing mix...\n")
        self.daw_text.see(tk.END)
        
        

    def _stop_mix(self):
        """Stop playback using sounddevice"""
        global is_playing
        try:
            sd.stop()
            is_playing = False
            self.daw_text.insert(tk.END, "⏹ Playback stopped\n")
            self.daw_text.see(tk.END)
            self._update_playback_status("Stopped")
        except Exception as e:
            print(f"Error stopping playback: {e}")
            is_playing = False
            self._update_playback_status("Error stopping")

    def _export_mix(self):
        """Export current mix to file"""
        global reverb_enabled, current_mix_with_reverb
        
        # Get current mix
        current_mix = self._create_current_mix()
        if not current_mix:
            messagebox.showwarning("No Mix", "No mix available to export")
            return
        
        # Check if reverb is enabled
        mix_to_export = current_mix
        suffix = ""
        if reverb_enabled and current_mix_with_reverb:
            mix_to_export = current_mix_with_reverb
            suffix = "_with_reverb"
        
        # Ask for filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        exports_dir = os.path.join(BASE_DIR, "exports")
        os.makedirs(exports_dir, exist_ok=True)
        default_name = os.path.join(exports_dir, f"custom_mix{suffix}_{timestamp}.mp3")
        
        filename = filedialog.asksaveasfilename(
            title="Save Mix As",
            defaultextension=".mp3",
            initialfile=default_name,
            filetypes=[("MP3 files", "*.mp3"), ("WAV files", "*.wav"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                if filename.endswith('.mp3'):
                    mix_to_export.export(filename, format="mp3", bitrate="192k")
                else:
                    mix_to_export.export(filename, format="wav")
                
                self.daw_text.insert(tk.END, f"💾 Mix exported to: {filename}\n")
                if reverb_enabled:
                    self.daw_text.insert(tk.END, "  (Includes reverb)\n")
                self.daw_text.see(tk.END)
                messagebox.showinfo("Export Complete", f"Mix exported successfully to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export mix: {e}")

    def _refresh_mix(self):
        """Refresh the mix with current settings"""
        global reverb_enabled
        reverb_enabled = False
        self.daw_text.insert(tk.END, "🔁 Mix refreshed with current settings\n")
        self.daw_text.insert(tk.END, "  (Reverb reset)\n")
        self.daw_text.see(tk.END)

    def _search_new_sound_for_track(self, track_index):
        """Search for new sound for a specific track with progress updates"""
        if track_index >= len(processed_tracks):
            return False
        
        track = processed_tracks[track_index]
        
        # Update progress in main thread
        def update_progress(text, value):
            if hasattr(self, '_replacement_status_label') and self._replacement_status_label:
                self._replacement_status_label.config(text=text)
                self._replacement_progress_var.set(value)
        
        self.root.after(0, lambda: update_progress("Getting track information...", 20))
        
        # Get the original tag for this track
        original_tag = None
        is_atmo = track.get('is_atmo', False)
        track_index_num = track.get('index', -1)
        
        # Find the corresponding tag
        if is_atmo:  # Atmo sound
            if scene_and_location_tags and len(scene_and_location_tags) > 0:
                original_tag = scene_and_location_tags[0] if isinstance(scene_and_location_tags, list) else scene_and_location_tags
        else:  # Object sound
            if track_index_num - 1 < len(recognized_tags):
                original_tag = recognized_tags[track_index_num - 1]
        
        if not original_tag:
            self.root.after(0, lambda: self.daw_text.insert(tk.END, f"❌ Could not find original tag for '{track['name']}'\n"))
            self.root.after(0, lambda: self.daw_text.see(tk.END))
            return False
        
        # Get current sound settings
        if not saved_sound_settings:
            self.root.after(0, lambda: self.daw_text.insert(tk.END, "❌ No sound settings found!\n"))
            self.root.after(0, lambda: self.daw_text.see(tk.END))
            return False
        
        settings = saved_sound_settings[-1]
        samplerate = settings['samplerate']
        duration_min = settings['duration_min']
        duration_max = settings['duration_max']
        license_type = settings['license_type']
        filetype = settings['filetype']
        quality_mode = settings['quality_mode']
        prefer_rating = settings['prefer_rating']
        
        # Get fallback method
        fallback_method = self.fallback_method_var.get() if hasattr(self, 'fallback_method_var') else "CSV"
        
        # Update progress
        self.root.after(0, lambda: update_progress(f"Searching for '{original_tag}'...", 40))
        
        # Search for new sound
        API_KEY = freesound_api_key
        
        # Create filterbank
        actual_samplerate = 44100 if samplerate == "any" else samplerate
        filterbank = None
        if AUDIO_AVAILABLE:
            try:
                filterbank = create_bark_filterbank(actual_samplerate)
            except Exception as e:
                print(f"Could not create filterbank: {e}")
        
        # Load hierarchy data
        hierarchy_data = []
        if fallback_method == 'CSV':
            csv_path = os.path.join(os.path.dirname(__file__), "tag_hierarchy.csv")
            hierarchy_data = load_tag_hierarchy(csv_path)
        
        sound = None
        current_tag = original_tag
        tried_tags = set()
        openai_fallback_tries = 0
        
        # Update progress
        self.root.after(0, lambda: update_progress("Checking sound availability...", 50))
        
        while current_tag and current_tag not in tried_tags:
            tried_tags.add(current_tag)
            
            # Define search tiers based on quality mode
            rating_steps = [5.0, 4.0, 3.5, 3.0, 2.5, 2.0, 0] if quality_mode and prefer_rating else [0]
            download_steps = [1000, 500, 100, 50, 0] if quality_mode and not prefer_rating else [0]
            
            found_valid_sound = False
            
            # Search through pages
            for round_num in range(1):  # Just check first page
                if found_valid_sound:
                    break
                
                for rating in rating_steps:
                    if found_valid_sound:
                        break
                    for downloads in download_steps:
                        if found_valid_sound:
                            break
                        
                        # Update progress
                        self.root.after(0, lambda: update_progress(f"Searching FreeSound database...", 60))
                        
                        # Search for sound
                        results = search_sounds(
                            API_KEY, current_tag, samplerate,
                            duration_min, duration_max, license_type, filetype, 
                            rating, page=1
                        )
                        
                        if results:
                            # Update progress
                            self.root.after(0, lambda: update_progress("Checking sound descriptions...", 70))
                            
                            # Find a sound not already used
                            for candidate_sound in results:
                                if candidate_sound['id'] in used_sound_ids:
                                    continue
                                
                                # Description check
                                if DESCRIPTION_CHECK_ENABLED and 'description' in candidate_sound:
                                    description = candidate_sound['description']
                                    if description and len(description.strip()) > 0:
                                        img_desc = image_input_description[0] if image_input_description else "No description"
                                        is_valid = check_description_with_openai(description, current_tag, img_desc)
                                        if not is_valid:
                                            continue
                                
                                # Found valid sound
                                sound = candidate_sound
                                found_valid_sound = True
                                break
                        
                        if found_valid_sound:
                            break
            
            if found_valid_sound:
                break
            else:
                # Try fallback
                if fallback_method == 'CSV':
                    current_tag = get_fallback_tag(current_tag, hierarchy_data)
                    if current_tag:
                        self.root.after(0, lambda: self.daw_text.insert(tk.END, f"  Trying fallback: '{current_tag}'\n"))
                        self.root.after(0, lambda: self.daw_text.see(tk.END))
                else:
                    openai_fallback_tries += 1
                    if openai_fallback_tries >= MAX_OPENAI_FALLBACK_TRIES:
                        current_tag = None
                    else:
                        current_tag = get_openai_fallback_tag(current_tag)
                        if current_tag:
                            self.root.after(0, lambda: self.daw_text.insert(tk.END, f"  OpenAI fallback: '{current_tag}'\n"))
                            self.root.after(0, lambda: self.daw_text.see(tk.END))
        
        if not sound:
            self.root.after(0, lambda: self.daw_text.insert(tk.END, f"❌ No new sound found for '{original_tag}'\n"))
            self.root.after(0, lambda: self.daw_text.see(tk.END))
            return False
        
        # Update progress
        self.root.after(0, lambda: update_progress("Downloading sound...", 80))
        
        # Get importance value
        object_importance = 0.5
        if not is_atmo and track_index_num >= 0:
            if 'importance_values' in globals() and importance_values and (track_index_num - 1) < len(importance_values):
                object_importance = importance_values[track_index_num - 1]
            else:
                object_importance = 0.7 - ((track_index_num - 1) * 0.1)
                object_importance = max(0.3, object_importance)
        
        # Update UI
        self.root.after(0, lambda: self.daw_text.insert(tk.END, f"\n🔍 Searching new sound for: {track['name']}\n"))
        self.root.after(0, lambda: self.daw_text.insert(tk.END, f"✓ Found: '{sound['name']}' for {current_tag}\n"))
        self.root.after(0, lambda: self.daw_text.see(tk.END))
        
        # Update progress
        self.root.after(0, lambda: update_progress("Processing audio...", 90))
        
        # Download and process the new sound
        if is_atmo:
            processed = download_and_process_sound(sound, f"atmo_{current_tag}", -1, actual_samplerate,
                                            importance_value=0.0, filterbank=filterbank)
        else:
            processed = download_and_process_sound(sound, f"{current_tag}_replacement", track_index_num,
                                            actual_samplerate, importance_value=object_importance, filterbank=filterbank)
        
        if not processed:
            self.root.after(0, lambda: self.daw_text.insert(tk.END, f"❌ Error processing new sound for '{original_tag}'\n"))
            self.root.after(0, lambda: self.daw_text.see(tk.END))
            return False
        
        # Update used sound IDs
        used_sound_ids.add(sound['id'])
        
        # Keep position and panning
        position = track['position']
        pan = track['pan']
        
        # Apply panning if not atmo
        if not is_atmo and pan != 0.0:
            processed = processed.pan(pan)
        
        # Update the track
        old_name = track['name']
        processed_tracks[track_index] = {
            'audio': processed,
            'name': sound['name'],
            'index': track_index_num,
            'position': position,
            'pan': pan,
            'is_atmo': is_atmo
        }
        
        # Update UI
        self.root.after(0, lambda: self.daw_text.insert(tk.END, f"✅ Replaced '{old_name}' with '{sound['name']}'\n"))
        self.root.after(0, lambda: self.daw_text.see(tk.END))
        
        # Stop playback if active
        global is_playing
        if is_playing:
            self.root.after(0, self._stop_mix)
            self.root.after(0, lambda: self.daw_text.insert(tk.END, "⏹ Playback stopped due to sound replacement\n"))
            self.root.after(0, lambda: self.daw_text.see(tk.END))
        
        return True
    
    def is_sound_processing_active(self):
        """Check if sound processing is currently active"""
        if hasattr(self, 'download_button'):
            return self.download_button.cget('state') == tk.DISABLED
        return False


    def _replace_and_refresh(self, track_index):
        """Replace sound and refresh DAW controls with progress bar"""
        # Create progress window
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Searching for New Sound...")
        progress_window.geometry("400x150")
        progress_window.transient(self.root)
        progress_window.grab_set()
        progress_window.protocol("WM_DELETE_WINDOW", lambda: None)  # Prevent closing
        
        progress_label = ttk.Label(progress_window, text=f"Searching for new sound...")
        progress_label.pack(pady=20)
        
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100)
        progress_bar.pack(pady=10, padx=20, fill=tk.X)
        
        status_label = ttk.Label(progress_window, text="Initializing...")
        status_label.pack(pady=10)
        
        # Store progress window reference
        self._replacement_progress_window = progress_window
        self._replacement_progress_var = progress_var
        self._replacement_status_label = status_label
        self._replacement_track_index = track_index
        
        # Create a thread for the replacement process
        def replace_sound_thread():
            try:
                # Update progress - must use root.after for thread safety
                def update_progress(text, value):
                    if self._replacement_progress_window and self._replacement_progress_window.winfo_exists():
                        self._replacement_status_label.config(text=text)
                        self._replacement_progress_var.set(value)
                
                # Stage 1: Loading settings
                self.root.after(0, lambda: update_progress("Loading settings...", 10))
                time.sleep(0.2)
                
                # Stage 2: Searching for sound
                self.root.after(0, lambda: update_progress("Searching for sound...", 30))
                
                # Call the actual replacement function
                success = self._search_new_sound_for_track(track_index)
                
                if success:
                    # Stage 6: Updating track
                    self.root.after(0, lambda: update_progress("Updating track...", 100))
                    time.sleep(0.2)
                    
                    # Update UI on main thread
                    self.root.after(0, lambda: self._on_replacement_complete(progress_window, track_index))
                else:
                    self.root.after(0, lambda: self._on_replacement_error("No suitable sound found", progress_window))
                
            except Exception as e:
                self.root.after(0, lambda: self._on_replacement_error(e, progress_window))
        
        # Start replacement thread
        replacement_thread = threading.Thread(target=replace_sound_thread, daemon=True)
        replacement_thread.start()
        
        # Start progress update loop
        self._update_replacement_progress(progress_window)

    def _on_replacement_complete(self, progress_window, track_index):
        """Callback when replacement is complete"""
        # Clear progress window references
        if hasattr(self, '_replacement_progress_window'):
            delattr(self, '_replacement_progress_window')
        if hasattr(self, '_replacement_progress_var'):
            delattr(self, '_replacement_progress_var')
        if hasattr(self, '_replacement_status_label'):
            delattr(self, '_replacement_status_label')
        
        # Close progress window
        if progress_window and progress_window.winfo_exists():
            progress_window.destroy()
        
        # Refresh DAW controls
        self.refresh_daw_controls()
        
        # Update log if text widget exists
        if hasattr(self, 'daw_text') and self.daw_text.winfo_exists():
            try:
                self.daw_text.insert(tk.END, f"✅ Sound replacement completed for track {track_index + 1}\n")
                self.daw_text.see(tk.END)
            except:
                pass
        
        self.update_status(f"Sound replaced successfully")
        self._update_daw_status(f"Track {track_index + 1} replaced")

    def _on_replacement_error(self, error, progress_window):
        """Callback when replacement fails"""
        # Clear progress window references
        if hasattr(self, '_replacement_progress_window'):
            delattr(self, '_replacement_progress_window')
        if hasattr(self, '_replacement_progress_var'):
            delattr(self, '_replacement_progress_var')
        if hasattr(self, '_replacement_status_label'):
            delattr(self, '_replacement_status_label')
        
        # Close progress window
        if progress_window and progress_window.winfo_exists():
            progress_window.destroy()
        
        error_msg = str(error)
        self.daw_text.insert(tk.END, f"❌ Error during sound replacement: {error_msg}\n")
        self.daw_text.see(tk.END)
        self.update_status("Error in sound replacement")
        
        messagebox.showerror("Replacement Error", f"Failed to replace sound:\n{error_msg}")

    def _update_replacement_progress(self, progress_window):
        """Update replacement progress window - keep it alive"""
        if progress_window and progress_window.winfo_exists():
            # Just keep the window responsive
            progress_window.after(100, lambda: self._update_replacement_progress(progress_window))
    
    def refresh_daw_controls(self):
        """Refresh the DAW controls with updated track information"""
        # Clear existing controls from the NEW container
        if hasattr(self, 'daw_controls_container'):
            for widget in self.daw_controls_container.winfo_children():
                widget.destroy()
        
        # Recreate controls with current track data
        if processed_tracks:
            self._create_daw_controls()
            
            # Update log if text widget exists
            if hasattr(self, 'daw_text') and self.daw_text.winfo_exists():
                try:
                    self.daw_text.insert(tk.END, "🔄 DAW controls refreshed with new sounds\n")
                    self.daw_text.see(tk.END)
                except:
                    pass
        
        self._update_daw_status("Controls refreshed")


    def _apply_reverb(self):
        """Apply reverb to the current mix"""
        global reverb_enabled
        
        # Double-check room detection using the helper method
        if not self._check_room_detection():
            messagebox.showwarning("No Room Detected", 
                                "Reverb cannot be applied because no room was detected in the image.")
            return
        
        # Call the instance method (you need to add this too)
        success = self._apply_reverb_instance()
        if success:
            self.daw_text.insert(tk.END, "🎚️ Reverb applied to mix\n")
            
            # Show the parameters used
            room_params = self._get_room_parameters()
            if room_params:
                self.daw_text.insert(tk.END, f"   Using: {', '.join(room_params)}\n")
            
            self.daw_text.see(tk.END)

    def _remove_reverb(self):
        """Remove reverb from the current mix"""
        global reverb_enabled
        
        # Call the instance method
        success = self._remove_reverb_instance()
        if success:
            self.daw_text.insert(tk.END, "🔇 Reverb removed from mix\n")
            self.daw_text.see(tk.END)

    def _apply_reverb_instance(self):
        """Instance method to apply reverb to the current mix"""
        global current_mix_raw, current_mix_with_reverb, reverb_enabled
        global room_size, damping, wet_level, width
        
        # Get current mix with volume adjustments
        current_mix = self._create_current_mix()
        if not current_mix:
            print("❌ No mix available to apply reverb")
            return False
        
        # Create progress window
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Applying Reverb...")
        progress_window.geometry("400x150")
        progress_window.transient(self.root)
        progress_window.grab_set()
        progress_window.protocol("WM_DELETE_WINDOW", lambda: None)  # Prevent closing
        
        progress_label = ttk.Label(progress_window, text="Applying reverb to mix...")
        progress_label.pack(pady=20)
        
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100)
        progress_bar.pack(pady=10, padx=20, fill=tk.X)
        
        status_label = ttk.Label(progress_window, text="Initializing...")
        status_label.pack(pady=10)
        
        # Update progress
        def update_progress(text, value):
            if progress_window.winfo_exists():
                status_label.config(text=text)
                progress_var.set(value)
                progress_window.update()
        
        try:
            # Stage 1: Initializing
            update_progress("Preparing audio mix...", 10)
            time.sleep(0.2)
            
            # Store raw mix WITH volume adjustments
            current_mix_raw = current_mix
            
            # Get reverb parameters from global room variables or use defaults
            room_size_val = room_size if 'room_size' in globals() and room_size is not None else 0.7
            damping_val = damping if 'damping' in globals() and damping is not None else 0.2
            wet_level_val = wet_level if 'wet_level' in globals() and wet_level is not None else 0.3
            width_val = width if 'width' in globals() and width is not None else 1.0
            
            # Stage 2: Setting parameters
            update_progress("Setting reverb parameters...", 30)
            time.sleep(0.2)
            
            print(f"Reverb parameters: Room Size={room_size_val:.2f}, "
                f"Damping={damping_val:.2f}, Wet Level={wet_level_val:.2f}, "
                f"Width={width_val:.2f}")
            
            # Stage 3: Processing reverb
            update_progress("Processing reverb effect...", 50)
            
            # Apply reverb
            current_mix_with_reverb = apply_reverb_to_audio(
                current_mix,
                room_size_val,
                damping_val,
                wet_level_val,
                width_val
            )
            
            # Stage 4: Finalizing
            update_progress("Finalizing...", 80)
            time.sleep(0.2)
            
            reverb_enabled = True
            
            # Stage 5: Complete
            update_progress("Reverb applied successfully!", 100)
            time.sleep(0.5)
            
            # Close progress window
            if progress_window.winfo_exists():
                progress_window.destroy()
            
            print("✅ Reverb applied")
            return True
            
        except Exception as e:
            if progress_window.winfo_exists():
                progress_window.destroy()
            print(f"❌ Error applying reverb: {e}")
            return False

    def _remove_reverb_instance(self):
        """Instance method to remove reverb and return to raw mix"""
        global reverb_enabled
        
        if not reverb_enabled:
            print("Reverb is not currently applied")
            return False
        
        # Create progress window
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Removing Reverb...")
        progress_window.geometry("400x150")
        progress_window.transient(self.root)
        progress_window.grab_set()
        progress_window.protocol("WM_DELETE_WINDOW", lambda: None)  # Prevent closing
        
        progress_label = ttk.Label(progress_window, text="Removing reverb from mix...")
        progress_label.pack(pady=20)
        
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100)
        progress_bar.pack(pady=10, padx=20, fill=tk.X)
        
        status_label = ttk.Label(progress_window, text="Initializing...")
        status_label.pack(pady=10)
        
        # Update progress
        def update_progress(text, value):
            if progress_window.winfo_exists():
                status_label.config(text=text)
                progress_var.set(value)
                progress_window.update()
        
        try:
            # Stage 1: Initializing
            update_progress("Preparing to remove reverb...", 20)
            time.sleep(0.2)
            
            # Stage 2: Removing reverb
            update_progress("Removing reverb effect...", 50)
            time.sleep(0.5)
            
            reverb_enabled = False
            
            # Stage 3: Finalizing
            update_progress("Reverb removed successfully!", 100)
            time.sleep(0.5)
            
            # Close progress window
            if progress_window.winfo_exists():
                progress_window.destroy()
            
            print("✅ Reverb removed")
            return True
            
        except Exception as e:
            if progress_window.winfo_exists():
                progress_window.destroy()
            print(f"❌ Error removing reverb: {e}")
            return False

    def _check_room_detection(self):
        """Check if room was detected and return True/False"""
        global room_detected
        
        # Check if room_detected exists in globals
        if 'room_detected' in globals():
            # Handle different formats
            if isinstance(room_detected, bool):
                return room_detected
            elif isinstance(room_detected, list) and len(room_detected) > 0:
                # Handle case where room_detected is a list from OpenAI
                value = room_detected[0]
                if isinstance(value, bool):
                    return value
                elif isinstance(value, str):
                    return value.lower() == 'true'
                else:
                    return False
            elif isinstance(room_detected, str):
                return room_detected.lower() in ['true', 'yes', '1']
        
        return False

    def _get_room_parameters(self):
        """Get room parameters as a list of strings for display"""
        params = []
        
        if 'room_size' in globals() and room_size is not None:
            if isinstance(room_size, list) and len(room_size) > 0:
                params.append(f"size: {room_size[0]}")
            else:
                params.append(f"size: {room_size}")
        
        if 'damping' in globals() and damping is not None:
            if isinstance(damping, list) and len(damping) > 0:
                params.append(f"damp: {damping[0]}")
            else:
                params.append(f"damp: {damping}")
        
        if 'wet_level' in globals() and wet_level is not None:
            if isinstance(wet_level, list) and len(wet_level) > 0:
                params.append(f"wet: {wet_level[0]}")
            else:
                params.append(f"wet: {wet_level}")
        
        if 'width' in globals() and width is not None:
            if isinstance(width, list) and len(width) > 0:
                params.append(f"width: {width[0]}")
            else:
                params.append(f"width: {width}")
        
        return params
    
    def _safe_create_daw_controls(self):
        """Safely create DAW controls with error handling"""
        try:
            # Check if widgets still exist
            if not self.daw_controls_frame or not self.daw_controls_frame.winfo_exists():
                print("DAW frame was destroyed, skipping control creation")
                return
                
            if not self.daw_text or not self.daw_text.winfo_exists():
                print("DAW text widget was destroyed, skipping control creation")
                return
            
            # Clear existing controls
            for widget in self.daw_controls_frame.winfo_children():
                if widget != self.daw_text:
                    try:
                        widget.destroy()
                    except:
                        pass
            
            # Create new controls
            self._create_daw_controls()
            
            # Final success message
            if self.daw_text and self.daw_text.winfo_exists():
                self.daw_text.insert(tk.END, "🎛️ DAW controls created successfully\n")
                self.daw_text.see(tk.END)
            
        except Exception as e:
            print(f"Could not create DAW controls: {e}")

    def check_and_restart_background_music(self):
        """Check current tab and restart background music if needed"""
        try:
            current_tab_index = self.notebook.index(self.notebook.select())
            if current_tab_index != 3:  # Not Sound Creation tab (index 3)
                start_background_music()
        except:
            pass

    def _clear_daw_controls(self):
        """Clear all DAW controls and show a placeholder message"""
        if hasattr(self, 'daw_controls_container') and self.daw_controls_container.winfo_exists():
            # Destroy all widgets in the container
            for widget in self.daw_controls_container.winfo_children():
                widget.destroy()
            
            # Clear volume sliders list
            if hasattr(self, 'volume_sliders'):
                self.volume_sliders = []
            
            # Add a placeholder message
            placeholder = tk.Label(self.daw_controls_container, 
                                text="No tracks available. Process an image first.",
                                bg="#ccffff",
                                font=("Arial", 10))
            placeholder.pack(pady=20)
        
        # Reset DAW status
        self._update_daw_status("Ready")
    
    def restart_application(self):
        """Restart the entire process by resetting all variables and GUI states"""
        # Ask for confirmation
        if not messagebox.askyesno("Restart Confirmation", 
                                "Are you sure you want to restart the process?\n\n"
                                "This will clear all current data and start fresh."):
            return
        
        stop_background_music()

        
        
        # Reset all global variables
        global IMAGE_FILE, uploaded_image, processed_image
        global recognized_tags, sound_pannings, scene_and_location_tags
        global image_input_description, importance_values, room_detected
        global room_size, damping, wet_level, width
        global saved_sound_settings, processed_tracks, used_sound_ids
        global downloaded_files, current_rating, reverb_enabled
        
        print("\n" + "="*60)
        print("DEBUG: BEFORE RESTART")
        print(f"recognized_tags: {len(recognized_tags)} tags: {recognized_tags}")
        print(f"processed_tracks: {len(processed_tracks)} tracks")
        print(f"used_sound_ids: {len(used_sound_ids)} IDs")
        print("="*60)


        # Reset image-related variables
        IMAGE_FILE = None
        uploaded_image = None
        processed_image = None
        
        # Reset detection results
        recognized_tags = []
        sound_pannings = []
        scene_and_location_tags = []
        image_input_description = []
        importance_values = []
        room_detected = None
        room_size = None
        damping = None
        wet_level = None
        width = None
        
        # Reset sound settings and tracks
        # saved_sound_settings = []
        processed_tracks = []
        used_sound_ids = set()
        downloaded_files = []
        
        # Reset audio state
        current_rating = 0
        reverb_enabled = False

        # Clear any audio analysis caches
        if 'filterbank' in globals():
            filterbank = None

         # Force garbage collection
        import gc
        gc.collect()
        
        # Clear file system - remove downloaded sound files
        try:
            for file in os.listdir('.'):
                if file.startswith('sound_') and file.endswith('.mp3'):
                    os.remove(file)
                if file.startswith('sound_mix_') or file.startswith('custom_mix_'):
                    os.remove(file)
                if file.startswith('playback_') and (file.endswith('.wav') or file.endswith('.mp3')):
                    os.remove(file)
        except Exception as e:
            print(f"Error cleaning up files: {e}")

        self.notebook.select(0)

        print("\nDEBUG: AFTER RESTART")
        print(f"recognized_tags: {len(recognized_tags)} tags: {recognized_tags}")
        print(f"processed_tracks: {len(processed_tracks)} tracks")
        print(f"used_sound_ids: {len(used_sound_ids)} IDs")
        print("="*60 + "\n")
        
        # Reset GUI elements
        # Tab 1 - Image tab
        self.image_label.config(image='', text="No image selected")
        self.generate_image_label.config(image='', text="No image generated yet")
        
        # Tab 2 - Object Detection tab
        if hasattr(self, 'results_text') and self.results_text.winfo_exists():
            self.results_text.delete(1.0, tk.END)
        
        '''
        # Tab 3 - Sound Settings tab (reset to defaults)
        self.quality_mode_var.set(True)
        self.prefer_rating_var.set(True)
        self.samplerate_var.set("any")
        self.duration_min_var.set(2)
        self.duration_max_var.set(10)
        self.license_var.set("any")
        self.filetype_var.set("any")
        '''
    
        # Tab 4 - Sound Creation tab
        if hasattr(self, 'daw_text') and self.daw_text.winfo_exists():
            self.daw_text.delete(1.0, tk.END)
        
        # Tab 4 - Image display
        if hasattr(self, 'tab4_image_label'):
            self.tab4_image_label.config(image='', text="No image loaded")
        
        if hasattr(self, 'daw_controls_container') and self.daw_controls_container.winfo_exists():
        # Destroy all widgets in the container
            for widget in self.daw_controls_container.winfo_children():
                try:
                    widget.destroy()
                except:
                    pass
            
            # Recreate the container with fresh widgets
            # Create a simple label indicating no tracks
            placeholder = tk.Label(self.daw_controls_container, 
                                text="No tracks available.",
                                bg="#ccffff",
                                font=("Arial", 10),
                                wraplength=400)
            placeholder.pack(pady=40, padx=20)

        if hasattr(self, 'daw_controls_container') and self.results_text.winfo_exists():
            self.daw_controls_container.grid_forget
            self.daw_controls_container.grid
        
        # Tab 4 - Reset DAW status
        if hasattr(self, 'daw_status_label'):
            self.daw_status_label.config(text="Ready")

        self.download_button.config(state=tk.NORMAL)
        
        # Tab 5 - Feedback tab
        self.set_rating(0)
        if hasattr(self, 'feedback_text') and self.feedback_text.winfo_exists():
            self.feedback_text.delete(1.0, tk.END)
        
        # Destroy and recreate DAW controls if they exist
        if hasattr(self, 'daw_controls_frame'):
            self.daw_controls_frame.grid_forget
            self.daw_controls_frame.grid

        self._clear_daw_controls
        
        # Switch to first tab
        self.notebook.select(0)
        
        # Reset status
        self.update_status("Application restarted - Ready for new image")
        
        
        
        # Clear any playback windows
        try:
            if hasattr(self, '_replacement_progress_window'):
                if self._replacement_progress_window.winfo_exists():
                    self._replacement_progress_window.destroy()
        except:
            pass
        
        messagebox.showinfo("Restart Complete", 
                        "Application has been reset successfully!\n\n"
                        "You can now start with a new image.")
        
        # Restart background music if not in Sound Creation tab
        self.root.after(500, self.check_and_restart_background_music)
        
    def _on_sounds_processed_safe(self, progress_window):
        """Thread-safe version - SIMPLIFIED"""
        def safe_cleanup():
            # Destroy progress window
            try:
                if progress_window and progress_window.winfo_exists():
                    progress_window.destroy()
            except:
                pass

            # Reset processing indicator
            if hasattr(self, 'processing_indicator'):
                self.processing_indicator.config(text="✓ Processing complete", fg=self.MODERN_SUCCESS)
                # Clear after 3 seconds
                self.root.after(3000, lambda: 
                    self.processing_indicator.config(text="", fg="#888888") 
                    if hasattr(self, 'processing_indicator') else None
                )
            
            # Enable download button
            if hasattr(self, 'download_button'):
                self.download_button.config(state=tk.NORMAL)
            
            # If we're currently in Sound Creation tab, fade out background music
            global current_active_tab
            if current_active_tab == "Sound Creation":
                global background_music_target_volume
                background_music_target_volume = 0.0
            
            # Update main status
            self.update_status("Sound processing completed")
            
            # Update text log if it exists
            if hasattr(self, 'daw_text') and self.daw_text and self.daw_text.winfo_exists():
                try:
                    if hasattr(self, 'enable_log_text'):
                        self.enable_log_text()
                    self.daw_text.insert(tk.END, "\n✓ Processing completed\n")
                    self.daw_text.see(tk.END)
                    if hasattr(self, 'disable_log_text'):
                        self.disable_log_text()
                except:
                    pass
            
            # Always try to create DAW controls (separate frame, bleibt erhalten)
            self.root.after(300, lambda: self._create_daw_controls())
        
        self.root.after(0, safe_cleanup)
    

    def _try_create_daw_controls(self):
        """Try to create DAW controls with safety checks - FIXED"""
        try:
            # Check if we're still in a valid state
            if not hasattr(self, 'daw_controls_container'):
                return
                
            # Check if container was destroyed
            if not self.daw_controls_container.winfo_exists():
                return
                
            # Use the create method directly
            self._create_daw_controls()
            
        except Exception as e:
            print(f"Could not create DAW controls: {e}")


    def update_tab4_image(self):
        """Update the image display in tab4"""
        global IMAGE_FILE
        
        if IMAGE_FILE and os.path.exists(IMAGE_FILE):
            try:
                # Load and resize image
                img = Image.open(IMAGE_FILE)
                # Resize to fit the left frame (max 300x300)
                img.thumbnail((280, 280))
                photo = ImageTk.PhotoImage(img)
                
                # Update the label
                self.tab4_image_label.config(image=photo)
                self.tab4_image_label.image = photo  # Keep a reference!
                self.tab4_image_label.config(text="")
            except Exception as e:
                print(f"Could not load image for tab4: {e}")
                self.tab4_image_label.config(text=f"Error loading image\n{str(e)[:30]}...")
        else:
            self.tab4_image_label.config(text="No image loaded")

    def _update_daw_status(self, message):
        """Update status label in DAW frame"""
        if hasattr(self, 'daw_status_label') and self.daw_status_label.winfo_exists():
            self.daw_status_label.config(text=message)

    def on_tab_changed(self, event=None):
        """Handle tab change events to control background music"""
        try:
            # Get current tab name
            current_tab_index = self.notebook.index(self.notebook.select())
            tab_names = ["Image", "Object Detection", "Sound Creation", "Feedback"]
            
            if current_tab_index < len(tab_names):
                current_tab = tab_names[current_tab_index]
                
                # Check if we're switching to Sound Creation during processing
                if current_tab == "Sound Creation" and self.is_sound_processing_active():
                    print("Switching to Sound Creation tab while processing - keeping music")
                    # Don't call toggle_background_music_for_tab since we want music to continue
                    global current_active_tab
                    current_active_tab = current_tab
                else:
                    toggle_background_music_for_tab(current_tab)
                    
        except Exception as e:
            print(f"Error handling tab change: {e}")
    
    def initialize_background_music(self):
        """Initialize and start background music"""
        # Check if music file exists
        music_path = os.path.join(BASE_DIR, background_music_file)
        
        if os.path.exists(music_path):
            try:
                # Get file info for debugging
                info = sf.info(music_path)
                print(f"Background music file: {music_path}")
                print(f"  Duration: {info.duration:.2f}s, Samplerate: {info.samplerate}Hz")
                
                # Set initial tab (Image tab)
                global current_active_tab
                current_active_tab = "Image"
                
                # Start background music
                start_background_music()
                
            except Exception as e:
                print(f"Error loading background music: {e}")
        else:
            print(f"⚠️ Background music file not found: {background_music_file}")


# ----------------------------------------------------------------------
# MAIN APPLICATION
# ----------------------------------------------------------------------

def main():
    """Main application entry point"""
    # Setup directories first - WICHTIG!
    print("="*60)
    print("Starting Image Extender v0.1.1")
    print("="*60)
    
    # Verzeichnisse erstellen
    global BASE_DIR
    BASE_DIR = setup_directories()
    
    # Check if we're in the right directory
    print(f"Working in directory: {BASE_DIR}")
    print("="*60)

    # start gui
    root = tk.Tk()

    def on_closing():
        print("Closing application...")
        stop_background_music()
        time.sleep(0.2)  # Give time for threads to stop
        root.destroy()
        print("Application closed")
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Set API key dialogs
    def set_api_key():
        global openai_api_key
        
        # Create a custom Toplevel window instead of using simpledialog
        dialog = tk.Toplevel(root)
        dialog.title("OpenAI API Key")
        
        # Make it modal and keep it on top
        dialog.transient(root)  # Associate with main window
        dialog.grab_set()  # Make it modal
        
        # Ensure it stays on top
        dialog.attributes('-topmost', True)
        
        # Position in center of screen
        dialog_width = 400
        dialog_height = 150
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width - dialog_width) // 2
        y = (screen_height - dialog_height) // 2
        dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")
        
        # Add widgets
        tk.Label(dialog, text="Enter your OpenAI API key:", 
                font=("Arial", 10)).pack(pady=10)
        
        key_entry = tk.Entry(dialog, width=40, show="*")
        key_entry.pack(pady=5, padx=20)
        key_entry.focus_set()  # Focus on entry field
        
        # Result variable
        result = {"key": None}
        
        def submit():
            result["key"] = key_entry.get()
            dialog.destroy()
        
        def cancel():
            dialog.destroy()
        
        # Buttons
        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=10)
        
        tk.Button(button_frame, text="Submit", command=submit, width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Cancel", command=cancel, width=10).pack(side=tk.LEFT, padx=5)
        
        # Bind Enter key to submit
        dialog.bind('<Return>', lambda e: submit())
        # Bind Escape key to cancel
        dialog.bind('<Escape>', lambda e: cancel())
        
        # Wait for dialog to close
        root.wait_window(dialog)
        
        # Process result
        if result["key"]:
            openai_api_key = result["key"]
            print("OpenAI API key set")
            return True
        return False
    
    def set_freesound_api_key():
        global freesound_api_key
        
        # Create a custom Toplevel window instead of using simpledialog
        dialog = tk.Toplevel(root)
        dialog.title("FreeSound API Key")
        
        # Make it modal and keep it on top
        dialog.transient(root)  # Associate with main window
        dialog.grab_set()  # Make it modal
        
        # Ensure it stays on top
        dialog.attributes('-topmost', True)
        
        # Position in center of screen
        dialog_width = 400
        dialog_height = 150
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width - dialog_width) // 2
        y = (screen_height - dialog_height) // 2
        dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")
        
        # Add widgets
        tk.Label(dialog, text="Enter your FreeSound API key:", 
                font=("Arial", 10)).pack(pady=10)
        
        key_entry = tk.Entry(dialog, width=40, show="*")
        key_entry.pack(pady=5, padx=20)
        key_entry.focus_set()  # Focus on entry field
        
        # Result variable
        result = {"key": None}
        
        def submit():
            result["key"] = key_entry.get()
            dialog.destroy()
        
        def cancel():
            dialog.destroy()
        
        # Buttons
        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=10)
        
        tk.Button(button_frame, text="Submit", command=submit, width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Cancel", command=cancel, width=10).pack(side=tk.LEFT, padx=5)
        
        # Bind Enter key to submit
        dialog.bind('<Return>', lambda e: submit())
        # Bind Escape key to cancel
        dialog.bind('<Escape>', lambda e: cancel())
        
        # Wait for dialog to close
        root.wait_window(dialog)
        
        # Process result
        if result["key"]:
            freesound_api_key = result["key"]
            print("FreeSound API key set")
            return True
        return False
    
    # Create menu
    menubar = tk.Menu(root)
    file_menu = tk.Menu(menubar, tearoff=0)
    file_menu.add_command(label="Set OpenAI API Key", command=set_api_key)
    file_menu.add_command(label="Set FreeSound API Key", command=set_freesound_api_key)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.quit)
    menubar.add_cascade(label="File", menu=file_menu)
    
    root.config(menu=menubar)
    
    # Create and run app
    app = ImageExtenderApp(root)
    
    # Check for API keys
    if not openai_api_key:
        set_api_key()
    if not freesound_api_key:
        set_freesound_api_key()
    
    root.mainloop()

if __name__ == "__main__":
    main()