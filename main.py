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
SUPERVISOR_PATH = os.path.join(settings.BASE_DIR, "models", "supervisor.pkl")

# --- STABILITY CONFIGURATION ---
LOOP_DELAY = 0.03 
STABILITY_WINDOW_SECONDS = 5.0
# Calculates buffer size needed to hold 5 seconds of predictions
STABILITY_BUFFER_SIZE = int(STABILITY_WINDOW_SECONDS / LOOP_DELAY) 
# Veto threshold: if 5s average is below 40%, ignore spikes
STABILITY_LOCK_THRESHOLD = 0.40 


def main():
    # 1. LOAD THE SUPERVISOR BRAIN
    print(f"Loading Supervisor Model from {SUPERVISOR_PATH}...")
    if not os.path.exists(SUPERVISOR_PATH):
        print("Error: Supervisor model not found! Run 'training/train_supervisor.py' first.")
        return
    
    supervisor = joblib.load(SUPERVISOR_PATH)
    print("Supervisor Loaded.")

    # 2. SETUP QUEUES (Need history for temporal features)
    results_fast = deque(maxlen=10)
    results_slow = deque(maxlen=10)
    results_exp = deque(maxlen=10)

    audio_feed_fast = queue.Queue()
    audio_feed_slow = queue.Queue()
    audio_feed_exp = queue.Queue()

    # NEW: Stability History for the Final Decision
    stability_queue = deque(maxlen=STABILITY_BUFFER_SIZE)

    # 3. START AI THREADS (Code remains as is)
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

    # 4. START AUDIO (FIXED FOR PASSTHROUGH & MUTING)
    print(f"Master Audio Stream started on Device {settings.AUDIO_DEVICE_INDEX}")
    # Shared state to let the main loop tell the audio thread to mute
    audio_state = {"mute": False}

    def master_callback(indata, outdata, frames, time, status):
        # A. Passthrough Logic (Play audio out to speakers)
        if audio_state["mute"]:
            outdata.fill(0) # Mute if ad detected
        else:
            outdata[:] = indata # Passthrough otherwise

        # B. Analysis Logic (Send to AI threads)
        mono = np.mean(indata, axis=1)
        audio_feed_fast.put(mono)
        audio_feed_slow.put(mono)
        audio_feed_exp.put(mono)

    # Use sd.Stream (Input & Output) instead of InputStream
    stream = sd.Stream(device=(settings.AUDIO_DEVICE_INDEX, None), 
                       channels=settings.CHANNELS, 
                       samplerate=settings.SAMPLE_RATE, 
                       callback=master_callback,
                       blocksize=int(settings.SAMPLE_RATE * 0.2))
    stream.start()


    # 5. VIDEO MANAGER
    video_mgr = VideoManager(CONTENT_FOLDER, settings.STREAM_DEVICE_INDEX)
    
    cv2.namedWindow("Live Ad Replacer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Ad Replacer", 800, 600)

    try:
        while True:
            # --- A. PREPARE FEATURES ---
            # 1. Snapshot History (Code remains as is)
            hist_fast = list(results_fast)
            hist_slow = list(results_slow)
            
            # 2. Get Raw Scores (Code remains as is)
            s1 = results_fast[-1] if len(results_fast) > 0 else 0.0
            s2 = results_slow[-1] if len(results_slow) > 0 else 0.0
            s3 = results_exp[-1]  if len(results_exp) > 0  else 0.0

            # 3. Calculate Temporal Features (Code remains as is)
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

            # Get Prediction (Probability of Ad)
            ad_probability = supervisor.predict_proba(input_data)[0][1]

            # --- C. STABILITY VETO CHECK (NEW) ---
            stability_queue.append(ad_probability)
            long_term_avg = sum(stability_queue) / len(stability_queue)
            
            final_decision_score = ad_probability
            
            # Apply VETO: If long-term context is low, ignore current spike
            if long_term_avg < STABILITY_LOCK_THRESHOLD:
                final_decision_score = 0.0 
            elif long_term_avg > (1.0 - STABILITY_LOCK_THRESHOLD):
                final_decision_score = 1.0
            # --- D. UPDATE AUDIO & VIDEO BASED ON STABLE SCORE ---
            is_ad_state = final_decision_score > settings.THRESHOLD
            
            # Mute/Passthrough based on the stable decision
            audio_state["mute"] = is_ad_state

            # Update the video manager
            frame = video_mgr.get_frame(is_ad_state)

            if frame is not None:
                # Debug Info: Draw the stable score and the average
                text_color = (0, 0, 255) if final_decision_score > settings.THRESHOLD else (0, 255, 0)
                
                cv2.putText(frame, 
                            f"Stable Prob: {final_decision_score:.2%} | Avg (5s): {long_term_avg:.2%}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                
                cv2.imshow("Live Ad Replacer", frame)

            # 5. Handle Exit
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