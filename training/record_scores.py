import time
import threading
import queue
import csv
import sounddevice as sd
import numpy as np
import keyboard # pip install keyboard
from collections import deque
import sys
import os

# Path hack to find your modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prediction.predictor import predict_live
import settings

def main():
    # Initialize Queues & Threads (Same as main.py)
    results_fast = deque(maxlen=1)
    results_slow = deque(maxlen=1)
    results_exp = deque(maxlen=1)
    
    audio_feed_fast = queue.Queue()
    audio_feed_slow = queue.Queue()
    audio_feed_exp = queue.Queue()

    # 3. START AI THREADS
    # Thread 1: Fast Sentry
    t1 = threading.Thread(target=predict_live, kwargs={
        "model_path": settings.MODEL_PATH, "chunk_duration": 1.0,
        "prediction_queue": results_fast, "input_audio_queue": audio_feed_fast
    }, daemon=True)
    t1.start()

    # Thread 2: Main Judge
    t2 = threading.Thread(target=predict_live, kwargs={
        "model_path": settings.MODEL_PATH2, "chunk_duration": 4.3,
        "prediction_queue": results_slow, "input_audio_queue": audio_feed_slow
    }, daemon=True)
    t2.start()

    # Thread 3: Experiment
    t3 = threading.Thread(target=predict_live, kwargs={
        "model_path": settings.MODEL_PATH2, "chunk_duration": 1.0,
        "prediction_queue": results_exp, "input_audio_queue": audio_feed_exp
    }, daemon=True)
    t3.start()

    # 4. START AUDIO CAPTURE
    print(f"Master Audio Stream started on Device {settings.DEVICE_INDEX}")
    
    def master_callback(indata, frames, time, status):
        # Mono mixdown
        mono = np.mean(indata, axis=1)
        # Feed all brains
        audio_feed_fast.put(mono)
        audio_feed_slow.put(mono)
        audio_feed_exp.put(mono)

    stream = sd.InputStream(device=settings.DEVICE_INDEX, channels=2, 
                            samplerate=settings.SAMPLE_RATE, callback=master_callback,
                            blocksize=int(settings.SAMPLE_RATE * 0.2))
    stream.start()

    print("RECORDING SCORES...")
    print("HOLD 'A' when an AD is playing.")
    print("HOLD 'C' when CONTENT (Game) is playing.")
    print("Press 'Q' to quit and save.")

    data_log = []

    try:
        while True:
            # Get current scores (default to 0)
            s1 = results_fast[-1] if results_fast else 0.0
            s2 = results_slow[-1] if results_slow else 0.0
            s3 = results_exp[-1]  if results_exp  else 0.0

            # Get User Label
            is_ad = 0
            is_content = 0
            
            if keyboard.is_pressed('a'): is_ad = 1
            if keyboard.is_pressed('c'): is_content = 1

            # Only record if user is actively pressing a key
            if is_ad or is_content:
                label = 1 if is_ad else 0
                data_log.append([s1, s2, s3, label])
                print(f"Logged: [{s1:.2f}, {s2:.2f}, {s3:.2f}] -> {label}")

            if keyboard.is_pressed('q'):
                break
            
            time.sleep(0.1)

    finally:
        # Save to CSV
        with open("dataset/score_data.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["fast", "slow", "exp", "label"])
            writer.writerows(data_log)
        print("Saved dataset/score_data.csv")

if __name__ == "__main__":
    main()
