import cv2
import time
import threading
import queue
import sounddevice as sd
import numpy as np
from collections import deque
from prediction.predictor import predict_live
import settings

def draw_status_bar(score, label, max_val=1.0, threshold=0.85):
    """
    Draws a single horizontal status bar for the dashboard.
    height: 120px, width: 600px
    """
    H, W = 120, 600
    img = np.zeros((H, W, 3), dtype=np.uint8)
    
    if score is None:
        cv2.putText(img, "Buffering...", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        # Draw border
        cv2.rectangle(img, (0, 0), (W-1, H-1), (50, 50, 50), 2)
        return img

    # Logic
    is_ad = score > threshold
    color = [0, 0, 255] if is_ad else [0, 255, 0]  # Red vs Green
    text_status = "ADVERTISEMENT" if is_ad else "CONTENT"
    
    # Draw Filled Bar
    # Normalize score to 0.0-1.0 range for width calculation
    fill_ratio = min(score / max_val, 1.0)
    fill_width = int(fill_ratio * W)
    
    # Background Bar (dimmed color)
    cv2.rectangle(img, (0, 0), (fill_width, H), color, -1)
    
    # Draw Text Info
    # Label (Top Left)
    cv2.putText(img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Status (Center Large)
    cv2.putText(img, text_status, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    
    # Score Details (Bottom Right)
    score_text = f"{score:.2f} / {max_val} (Thresh: {threshold})"
    cv2.putText(img, score_text, (W - 280, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Draw border for separation
    cv2.rectangle(img, (0, 0), (W-1, H-1), (50, 50, 50), 2)
    
    return img

def main():
    # 1. SETUP QUEUES
    results_fast = deque(maxlen=1)
    results_slow = deque(maxlen=1)
    results_exp = deque(maxlen=1)

    audio_feed_fast = queue.Queue()
    audio_feed_slow = queue.Queue()
    audio_feed_exp = queue.Queue()

    # 2. START THREADS
    # Thread 1: Fast Sentry (1.0s)
    t1 = threading.Thread(target=predict_live, kwargs={
        "model_path": settings.MODEL_PATH, "chunk_duration": 1.0,
        "prediction_queue": results_fast, "input_audio_queue": audio_feed_fast, "show_hud": False
    }, daemon=True)
    t1.start()

    # Thread 2: Main Judge (4.3s)
    t2 = threading.Thread(target=predict_live, kwargs={
        "model_path": settings.MODEL_PATH2, "chunk_duration": 4.3,
        "prediction_queue": results_slow, "input_audio_queue": audio_feed_slow, "show_hud": False
    }, daemon=True)
    t2.start()

    # Thread 3: Experiment (Main @ 1.0s)
    t3 = threading.Thread(target=predict_live, kwargs={
        "model_path": settings.MODEL_PATH2, "chunk_duration": 1.0,
        "prediction_queue": results_exp, "input_audio_queue": audio_feed_exp, "show_hud": False
    }, daemon=True)
    t3.start()

    # 3. START AUDIO
    print(f"Master Stream started on Device {settings.DEVICE_INDEX}")
    def master_callback(indata, frames, time, status):
        mono = np.mean(indata, axis=1)
        audio_feed_fast.put(mono)
        audio_feed_slow.put(mono)
        audio_feed_exp.put(mono)

    stream = sd.InputStream(device=settings.DEVICE_INDEX, channels=2, 
                            samplerate=settings.SAMPLE_RATE, callback=master_callback,
                            blocksize=int(settings.SAMPLE_RATE * 0.2))
    stream.start()

    # 4. DASHBOARD SETUP
    cv2.namedWindow("Ad Detector 5-Row Dashboard", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Ad Detector 5-Row Dashboard", 600, 600)

    # Init scores
    s_fast = 0.0
    s_slow = 0.0
    s_exp = 0.0

    try:
        while True:
            # Update latest scores
            if len(results_fast) > 0: s_fast = results_fast[-1]
            if len(results_slow) > 0: s_slow = results_slow[-1]
            if len(results_exp) > 0: s_exp = results_exp[-1]

            # Calculate Sums
            sum_2 = s_fast + s_slow
            sum_3 = s_fast + s_slow + s_exp

            # Generate 5 Panels
            # Row 1: Fast Sentry (Max 1.0, Thresh 0.85)
            p1 = draw_status_bar(s_fast, "1. Fast Sentry (1.0s)", max_val=1.0, threshold=0.85)
            
            # Row 2: Main Judge (Max 1.0, Thresh 0.85)
            p2 = draw_status_bar(s_slow, "2. Main Judge (4.3s)", max_val=1.0, threshold=0.85)
            
            # Row 3: Experiment (Max 1.0, Thresh 0.85)
            p3 = draw_status_bar(s_exp, "3. Experiment (Main @ 1.0s)", max_val=1.0, threshold=0.85)
            
            # Row 4: Sum of First Two (Max 2.0, Thresh 0.70)
            p4 = draw_status_bar(sum_2, "4. Sum (Fast + Main)", max_val=2.0, threshold=0.70)
            
            # Row 5: Sum of All Three (Max 3.0, Thresh 1.00)
            p5 = draw_status_bar(sum_3, "5. Sum (All Three)", max_val=3.0, threshold=1.00)

            # Stack Vertically
            dashboard = np.vstack((p1, p2, p3, p4, p5))

            # Show
            cv2.imshow("Ad Detector 5-Row Dashboard", dashboard)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stream.stop()
        stream.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
