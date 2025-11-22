import cv2
import time
import os
import random
import collections
import numpy as np
import sounddevice as sd
import settings
from prediction.predictor import AdPredictor

CONTENT_FOLDER = "content"
STREAM_DEVICE_INDEX = 0  # Camera Index for OBS Video
AUDIO_DEVICE_INDEX = 1 # Set to your Virtual Cable Output index if needed, else None for default

def main():
    # 1. Initialize Predictor
    print("Loading model...")
    predictor = AdPredictor(settings.MODEL_PATH)
    
    # 2. Setup Audio Bridge
    # This queue holds audio chunks moving from the background audio thread to the main loop
    audio_queue = collections.deque()
    state = {"score": 0.0}

    def audio_callback(indata, outdata, frames, time_info, status):
        if status:
            print(status)
        # Mix to mono and add to queue
        mono = np.mean(indata, axis=1)
        audio_queue.extend(mono)

        if state["score"] > 0.85:
            outdata.fill(0)
        else:
            outdata[:] = indata

    # 3. Load content files
    content_files = [os.path.join(CONTENT_FOLDER, f) 
                     for f in os.listdir(CONTENT_FOLDER)
                     if f.lower().endswith((".png", ".jpg", ".jpeg", ".mp4", ".avi"))]
    
    # 4. Setup Video Capture
    cap_passthrough = cv2.VideoCapture(STREAM_DEVICE_INDEX)
    cap_passthrough.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap_passthrough.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    
    if not cap_passthrough.isOpened():
        print("Error: Could not open video device.")
        return

    cv2.namedWindow("Main Display", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Main Display", 800, 600)

    ad_cap = None # State for video ad playback

    print("Starting main loop...")
    # Start non-blocking audio stream
    try:
        with sd.Stream(device=(AUDIO_DEVICE_INDEX, None), channels=1, samplerate=44100, callback=audio_callback):
            score = 0.0
            while True:
                # --- A. Get Video Frame ---
                ret_video, frame = cap_passthrough.read()
                if not ret_video:
                    print("Video stream lost.")
                    break
                
                current_content = frame

                # --- B. Get Audio & Predict ---
                # Consume all available audio in the queue to stay real-time
                if len(audio_queue) > 0:
                    chunk = []
                    while len(audio_queue) > 0:
                        chunk.append(audio_queue.popleft())
                    
                    # Get immediate prediction score
                    score = predictor.predict(np.array(chunk))
                    state["score"] = score

                # --- C. Content Switching Logic ---
                if score > 0.85 and content_files:
                    # Ad Detected
                    if ad_cap is None:
                        # Pick new content
                        f = random.choice(content_files)
                        if f.endswith((".png", ".jpg")):
                            img = cv2.imread(f)
                            if img is not None: current_content = cv2.resize(img, (800, 600))
                        else:
                            ad_cap = cv2.VideoCapture(f)
                    
                    if ad_cap is not None:
                        ret_ad, frame_ad = ad_cap.read()
                        if ret_ad:
                            current_content = cv2.resize(frame_ad, (800, 600))
                        else:
                            ad_cap.release()
                            ad_cap = None
                else:
                    # Content Detected (Passthrough)
                    if ad_cap is not None:
                        ad_cap.release()
                        ad_cap = None
                    
                    # Overlay
                    cv2.putText(current_content, f"LIVE ({score:.1%})", (50, 550), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                cv2.putText(current_content, f"Confidence: {state['score']:.2%}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.imshow("Main Display", current_content)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except Exception as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cap_passthrough.release()
        if ad_cap: ad_cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()