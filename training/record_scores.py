import time
import threading
import queue
import csv
import sounddevice as sd
import numpy as np
import keyboard 
from collections import deque
import sys
import os

# Path hack
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prediction.predictor import predict_live
import settings

def main():
    # 1. SETUP QUEUES
    results_fast = deque(maxlen=10)
    results_slow = deque(maxlen=10)
    results_exp = deque(maxlen=10)
    
    audio_feed_fast = queue.Queue()
    audio_feed_slow = queue.Queue()
    audio_feed_exp = queue.Queue()

    # 2. START THREADS
    t1 = threading.Thread(target=predict_live, kwargs={
        "model_path": settings.MODEL_PATH, "thread_name": "Fast", "chunk_duration": 1.0,
        "prediction_queue": results_fast, "input_audio_queue": audio_feed_fast
    }, daemon=True)
    t1.start()

    t2 = threading.Thread(target=predict_live, kwargs={
        "model_path": settings.MODEL_PATH2, "thread_name": "Main", "chunk_duration": 4.3,
        "prediction_queue": results_slow, "input_audio_queue": audio_feed_slow
    }, daemon=True)
    t2.start()

    t3 = threading.Thread(target=predict_live, kwargs={
        "model_path": settings.MODEL_PATH2, "thread_name": "Experimental", "chunk_duration": 1.0,
        "prediction_queue": results_exp, "input_audio_queue": audio_feed_exp
    }, daemon=True)
    t3.start()

    # 3. START AUDIO
    print(f"Master Audio Stream started on Device {settings.DEVICE_INDEX}")
    def master_callback(indata, frames, time, status):
        mono = np.mean(indata, axis=1)
        audio_feed_fast.put(mono)
        audio_feed_slow.put(mono)
        audio_feed_exp.put(mono)

    stream = sd.InputStream(device=settings.DEVICE_INDEX, channels=2, 
                            samplerate=settings.SAMPLE_RATE, callback=master_callback,
                            blocksize=int(settings.SAMPLE_RATE * 0.2))
    stream.start()

    print("RECORDING SCORES (HEADLESS)...")
    print("HOLD 'A' for Ad, 'C' for Content.")
    print("Press 'Q' to quit.")
    
    data_log = []

    try:
        while True:
            # Get latest scores (default to 0.0)
            s1 = results_fast[-1] if len(results_fast) > 0 else 0.0
            s2 = results_slow[-1] if len(results_slow) > 0 else 0.0
            s3 = results_exp[-1]  if len(results_exp) > 0  else 0.0

            # --- TEMPORAL FEATURE CALCULATION ---
            # 1. Convert deques to lists for math
            hist_fast = list(results_fast)
            hist_slow = list(results_slow)

            # 2. Calculate Features (Handle empty startup case)
            if len(hist_fast) > 1:
                fast_avg = np.mean(hist_fast)       # Trend
                fast_std = np.std(hist_fast)        # Stability/Panic
                fast_delta = hist_fast[-1] - hist_fast[0] # Velocity
            else:
                fast_avg, fast_std, fast_delta = s1, 0.0, 0.0

            if len(hist_slow) > 1:
                slow_delta = hist_slow[-1] - hist_slow[0] # Is confidence rising?
            else:
                slow_delta = 0.0
            # ------------------------------------

            # User Label Input
            is_ad = 0
            is_content = 0
            if keyboard.is_pressed('a'): is_ad = 1
            if keyboard.is_pressed('c'): is_content = 1

            if is_ad or is_content:
                label = 1 if is_ad else 0
                
                # SAVE ALL FEATURES
                data_row = [s1, s2, s3, fast_avg, fast_std, fast_delta, slow_delta, label]
                data_log.append(data_row)
                
                # Optional: Print status so you know it's working
                sys.stdout.write(f"\rLogged {len(data_log)} samples... (Last: {label})")
                sys.stdout.flush()

            if keyboard.is_pressed('q'):
                print("\nQuitting...")
                break
            
            time.sleep(0.05)

    finally:
        stream.stop()
        stream.close()
        
        # Save to CSV (APPEND MODE)
        if len(data_log) > 0:
            csv_path = os.path.join(settings.BASE_DIR, "dataset", "score_data.csv")
            
            # Check if file exists so we don't write headers twice
            file_exists = os.path.isfile(csv_path)
            
            # Open in 'a' (Append) mode instead of 'w' (Write)
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                
                # Only write headers if this is a brand new file
                if not file_exists:
                    headers = ["fast", "slow", "exp", "fast_avg", "fast_std", "fast_delta", "slow_delta", "label"]
                    writer.writerow(headers)
                
                writer.writerows(data_log)
                
            print(f"Appended {len(data_log)} rows to {csv_path}")
        else:
            print("No data recorded.")

if __name__ == "__main__":
    main()
