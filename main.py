import time
import threading
import queue
import cv2
import sounddevice as sd
import numpy as np
import pandas as pd 
import joblib 
import os
from collections import deque

# Import Modules
from prediction.predictor import predict_live
from display.video_manager import VideoManager
import settings

# --- CONFIGURATION ---
CONTENT_FOLDER = "content"
STREAM_DEVICE_INDEX = 1
SUPERVISOR_PATH = os.path.join(settings.BASE_DIR, "models", "supervisor.pkl")

# Stability Config
STABILITY_BUFFER_SIZE = int(5.0 / 0.03) # 5 seconds / loop delay (approx)
STABILITY_LOCK_THRESHOLD = 0.40

def main():
    # 1. LOAD THE SUPERVISOR BRAIN
    print(f"Loading Supervisor Model from {SUPERVISOR_PATH}...")
    if not os.path.exists(SUPERVISOR_PATH):
        print("Error: Supervisor model not found! Run 'training/train_supervisor.py' first.")
        return
    
    supervisor = joblib.load(SUPERVISOR_PATH)
    print("✅ Supervisor Loaded.")

    # 2. SETUP QUEUES
    results_fast = deque(maxlen=10)
    results_slow = deque(maxlen=10)
    results_exp = deque(maxlen=10)
    
    # NEW: Stability History for the Final Decision
    stability_queue = deque(maxlen=STABILITY_BUFFER_SIZE)

    audio_feed_fast = queue.Queue()
    audio_feed_slow = queue.Queue()
    audio_feed_exp = queue.Queue()

    # 3. START AI THREADS
    t1 = threading.Thread(target=predict_live, kwargs={
        "model_path": settings.MODEL_PATH, "chunk_duration": 1.0,
        "prediction_queue": results_fast, "input_audio_queue": audio_feed_fast
    }, daemon=True)
    t1.start()

    t2 = threading.Thread(target=predict_live, kwargs={
        "model_path": settings.MODEL_PATH2, "chunk_duration": 4.3,
        "prediction_queue": results_slow, "input_audio_queue": audio_feed_slow
    }, daemon=True)
    t2.start()

    t3 = threading.Thread(target=predict_live, kwargs={
        "model_path": settings.MODEL_PATH2, "chunk_duration": 1.0,
        "prediction_queue": results_exp, "input_audio_queue": audio_feed_exp
    }, daemon=True)
    t3.start()

    # 4. START AUDIO
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

    # 5. VIDEO MANAGER
    video_mgr = VideoManager(CONTENT_FOLDER, STREAM_DEVICE_INDEX)
    
    cv2.namedWindow("Live Ad Replacer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Ad Replacer", 800, 600)

    try:
        while True:
            # --- A. PREPARE FEATURES ---
            hist_fast = list(results_fast)
            hist_slow = list(results_slow)
            
            s1 = results_fast[-1] if len(results_fast) > 0 else 0.0
            s2 = results_slow[-1] if len(results_slow) > 0 else 0.0
            s3 = results_exp[-1]  if len(results_exp) > 0  else 0.0

            if len(hist_fast) > 1:
                fast_avg = np.mean(hist_fast)
                fast_std = np.std(hist_fast)
                fast_delta = hist_fast[-1] - hist_fast[0]
            else:
                fast_avg, fast_std, fast_delta = s1, 0.0, 0.0

            if len(hist_slow) > 1:
                slow_delta = hist_slow[-1] - hist_slow[0]
            else:
                slow_delta = 0.0

            # --- B. ASK THE SUPERVISOR ---
            input_data = pd.DataFrame([[
                s1, s2, s3, 
                fast_avg, fast_std, fast_delta, 
                slow_delta
            ]], columns=["fast", "slow", "exp", "fast_avg", "fast_std", "fast_delta", "slow_delta"])

            # Raw Probability from Supervisor (0.0 to 1.0)
            ad_probability = supervisor.predict_proba(input_data)[0][1]
            
            # --- C. STABILITY CHECK (NEW) ---
            # 1. Add current guess to history
            stability_queue.append(ad_probability)
            
            # 2. Calculate 5-second average
            if len(stability_queue) > 0:
                long_term_avg = sum(stability_queue) / len(stability_queue)
            else:
                long_term_avg = 0.0
            
            # 3. The Decision
            # If the long-term average is too low (Content), force the probability down
            # This prevents a 1-second spike from triggering the ad logic
            final_decision_score = ad_probability
            
            if long_term_avg < STABILITY_LOCK_THRESHOLD:
                # VETO: The system is stable on "Content", ignore spikes
                final_decision_score = 0.0 
            
            # --- D. UPDATE VIDEO ---
            frame = video_mgr.get_frame(final_decision_score, threshold=0.5)

            if frame is not None:
                # Debug Info
                text_color = (0, 0, 255) if final_decision_score > 0.5 else (0, 255, 0)
                cv2.putText(frame, f"Prob: {ad_probability:.2f} | Avg over {STABILITY_BUFFER_SIZE:.1f}s: {long_term_avg:.2f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                
                cv2.imshow("Live Ad Replacer", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
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
