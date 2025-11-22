import time
import threading
import queue
import sounddevice as sd
import numpy as np
from collections import deque
from prediction.predictor import predict_live
import prediction.visualizer as viz  # <--- Import your new module
import settings

def main():
    # 1. SETUP QUEUES
    results_fast = deque(maxlen=1)
    results_slow = deque(maxlen=1)
    results_exp = deque(maxlen=1)

    audio_feed_fast = queue.Queue()
    audio_feed_slow = queue.Queue()
    audio_feed_exp = queue.Queue()

    # 2. START THREADS
    # Thread 1: Fast
    t1 = threading.Thread(target=predict_live, kwargs={
        "model_path": settings.MODEL_PATH, "chunk_duration": 1.0,
        "prediction_queue": results_fast, "input_audio_queue": audio_feed_fast
    }, daemon=True)
    t1.start()

    # Thread 2: Slow
    t2 = threading.Thread(target=predict_live, kwargs={
        "model_path": settings.MODEL_PATH2, "chunk_duration": 4.3,
        "prediction_queue": results_slow, "input_audio_queue": audio_feed_slow
    }, daemon=True)
    t2.start()

    # Thread 3: Exp
    t3 = threading.Thread(target=predict_live, kwargs={
        "model_path": settings.MODEL_PATH2, "chunk_duration": 1.0,
        "prediction_queue": results_exp, "input_audio_queue": audio_feed_exp
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

    # 4. UI LOOP
    viz.init_dashboard()

    try:
        while True:
            # Collect scores
            scores = {
                "fast": results_fast[-1] if results_fast else None,
                "slow": results_slow[-1] if results_slow else None,
                "exp":  results_exp[-1]  if results_exp  else None
            }

            # Update UI (Returns True if 'q' pressed)
            should_quit = viz.update_dashboard(scores)
            
            if should_quit:
                break
            
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stream.stop()
        stream.close()
        viz.close_dashboard()

if __name__ == "__main__":
    main()
