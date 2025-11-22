import os
import librosa
import soundfile as sf
import numpy as np
import sys

# --- PATH to find settings.py ---
# 1. Get the folder where this script lives (.../LiveSwap/training)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Go up one level to the Project Root (.../LiveSwap)
project_root = os.path.dirname(current_dir)

# 3. Add the root to Python's path
sys.path.append(project_root)

# Direcotry paths with recording
from settings import RAW_RECORDINGS_DIR as RAW_FOLDER
from settings import PROCESSED_DATASET_DIR as OUTPUT_BASE

# Two Strategies:
# 1. Main Model: 4.3s clips (High Accuracy)
# 2. Fast Model: 1.0s clips (High Speed/Reflex) with heavy overlap (0.5s stride)
SLICE_CONFIGS = [
    {"name": "main_4.3s", "duration": 4.3, "stride": 4.0}, 
    {"name": "fast_1.0s", "duration": 1.0, "stride": 0.5} 
]

def slice_audio_files():
    print(f"Slicing audio from: {RAW_FOLDER}")
    
    # Process both "ads" and "content" folders
    for category in ["ads", "content"]:
        raw_path = os.path.join(RAW_FOLDER, category)
        
        # Skip if folder doesn't exist (e.g., if you only recorded content so far)
        if not os.path.exists(raw_path):
            print(f"Skipping '{category}' (Folder not found). Go record some!")
            continue

        files = [f for f in os.listdir(raw_path) if f.endswith(".wav")]
        if not files:
            print(f"Folder '{category}' is empty.")
            continue

        for filename in files:
            file_path = os.path.join(raw_path, filename)
            print(f"   Processing: {filename}...")

            # 1. Load the full audio file
            # sr=None preserves the original sampling rate (44100)
            try:
                audio, sr = librosa.load(file_path, sr=None)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
            
            # 2. Apply Both Slicing Strategies
            for config in SLICE_CONFIGS:
                duration_samples = int(config["duration"] * sr)
                stride_samples = int(config["stride"] * sr)
                
                # Create output folder (e.g., dataset/main_4.3s/content/)
                out_folder = os.path.join(OUTPUT_BASE, config["name"], category)
                os.makedirs(out_folder, exist_ok=True)
                
                clips_saved = 0
                
                # 3. Sliding Window Loop
                for start in range(0, len(audio) - duration_samples, stride_samples):
                    chunk = audio[start : start + duration_samples]
                    
                    # Validation: Check for Silence
                    # If the max volume is near 0, it's probably a mute gap. Skip it.
                    if np.max(np.abs(chunk)) < 0.005:
                        continue

                    # Save the slice
                    # Name format: rawfilename_001.wav
                    out_name = f"{filename[:-4]}_{clips_saved:04d}.wav"
                    sf.write(os.path.join(out_folder, out_name), chunk, sr)
                    clips_saved += 1
                
                print(f"      -> Generated {clips_saved} clips for {config['name']}")

if __name__ == "__main__":
    slice_audio_files()
