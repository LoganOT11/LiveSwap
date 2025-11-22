import argparse
import sounddevice as sd
import soundfile as sf
import numpy as np
import queue
import sys
import os
from datetime import datetime

# --- PATH to find settings.py ---
# 1. Get the folder where this script lives (.../LiveSwap/training)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Go up one level to the Project Root (.../LiveSwap)
project_root = os.path.dirname(current_dir)

# 3. Add the root to Python's path
sys.path.append(project_root)

from settings import AUDIO_DEVICE_INDEX, SAMPLE_RATE, CHANNELS

# High-performance Queue
# This holds audio in RAM temporarily so the recording process doesn't lag
q = queue.Queue()

def callback(indata, frames, time, status):
    """
    This function runs in a background thread.
    It grabs audio from the VB-Cable and throws it into the Queue.
    """
    if status:
        print(f"Audio Status: {status}", file=sys.stderr)
    
    # We must copy the data, otherwise sounddevice overwrites it in the next frame
    q.put(indata.copy())

def record_audio(label):
    # Setup output directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "dataset", "raw_recordings", label)
    os.makedirs(output_dir, exist_ok=True)

    # Filename Creation: ads_2023-11-21_14-30-01.wav
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{label}_{timestamp}.wav"
    filepath = os.path.join(output_dir, filename)

    print(f"\n" + "="*40)
    print(f"RECORDING CLASS: [{label.upper()}]")
    print(f"Saving to: {filepath}")
    print(f"Device Index: {AUDIO_DEVICE_INDEX}")
    print(f"Press Ctrl+C to STOP recording")
    print("="*40 + "\n")

    # Run File and Start Stream
    try:
        # Open a SoundFile for writing (streaming mode)
        with sf.SoundFile(filepath, mode='x', samplerate=SAMPLE_RATE,
                          channels=CHANNELS, subtype='PCM_24') as file:
            
            # Start the microphone listener
            with sd.InputStream(samplerate=SAMPLE_RATE, device=AUDIO_DEVICE_INDEX,
                                channels=CHANNELS, callback=callback):
                print("Recording... (Audio is being captured)")
                while True:
                    # Pull audio from Queue and write to Disk
                    file.write(q.get())

    except KeyboardInterrupt:
        print(f"\n\nRecording Stopped.")
        print(f"File saved successfully: {filename}")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    # Pass argument for type of recording: ads or content
    parser = argparse.ArgumentParser(description="Sports Stream Data Recorder")
    parser.add_argument("--label", required=True, choices=["ads", "content"],
                        help="Specify what you are recording: 'ads' or 'content'")
    
    args = parser.parse_args()
    
    # Run the recorder
    record_audio(args.label)
