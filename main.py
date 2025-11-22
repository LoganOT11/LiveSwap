import time
import threading
import queue
import cv2
import sounddevice as sd
import numpy as np
from collections import deque

# Import your modules
from prediction.predictor import predict_live
from display.video_manager import VideoManager
import settings

# --- CONFIGURATION ---
CONTENT_FOLDER = "content"
STREAM_DEVICE_INDEX = 1 # Index for OBS Virtual Camera / Capture Card
AD_THRESHOLD_SUM = 1.0  # Trigger Ad if Sum(Thread1 + Thread2 + Thread3) > 1.0

def main():
    # 1. SETUP QUEUES (AI Output)
    results_fast = deque(maxlen=1)
    results_slow = deque(maxlen=1)
    results_exp = deque(maxlen=1)

    # 2. AUDIO QUEUES (AI Input)
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

    # 5. INITIALIZE VIDEO MANAGER (The Display)
    video_mgr = VideoManager(CONTENT_FOLDER, STREAM_DEVICE_INDEX)
    
    # Create Window
    cv2.namedWindow("Live Ad Replacer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Ad Replacer", 800, 600)

    try:
        while True:
            # --- A. GET SCORES ---
            # Default to 0.0 if threads haven't reported yet
            s1 = results_fast[-1] if len(results_fast) > 0 else 0.0
            s2 = results_slow[-1] if len(results_slow) > 0 else 0.0
            s3 = results_exp[-1]  if len(results_exp) > 0  else 0.0

            # --- B. CALCULATE TOTAL SUM ---
            total_score = s1 + s2 + s3

            # --- C. GET FRAME ---
            # VideoManager logic:
            # If total_score > 1.0 -> Return Ad Frame
            # Else -> Return Live Game Frame
            frame = video_mgr.get_frame(total_score, threshold=AD_THRESHOLD_SUM)

            # --- D. DISPLAY ---
            if frame is not None:
                cv2.imshow("Live Ad Replacer", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Run loop at approx 30 FPS
            time.sleep(0.03)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stream.stop()
        stream.close()
        video_mgr.release()
        cv2.destroyAllWindows()
        print("Cleanup complete.")

if __name__ == "__main__":
    main()
