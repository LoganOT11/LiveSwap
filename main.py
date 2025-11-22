import cv2
import time
import threading
import os
import random
from collections import deque
from prediction.predictor import predict_live
import settings
import numpy as np

CONTENT_FOLDER = "content"
STREAM_DEVICE_INDEX = 0 

def main():
    # Queue to receive prediction values
    prediction_queue = deque(maxlen=1)

    # --- Start predictor in background thread ---
    predictor_thread = threading.Thread(
        target=predict_live,
        kwargs={
            "model_path": settings.MODEL_PATH,
            "show_hud": False,
            "prediction_queue": prediction_queue
        },
        daemon=True
    )
    predictor_thread.start()

    # --- Load content files ---
    content_files = [os.path.join(CONTENT_FOLDER, f) 
                     for f in os.listdir(CONTENT_FOLDER)
                     if f.lower().endswith((".png", ".jpg", ".jpeg", ".mp4", ".avi"))]
    if not content_files:
        print(f"No content files found in {CONTENT_FOLDER}. Ads will display a blank screen.")
        
    # --- Video Passthrough Setup ---
    CAP_WIDTH, CAP_HEIGHT = 800, 600
    cap_passthrough = None
    
    try:
        # **Using the confirmed index 0 for the OBS Virtual Camera**
        cap_passthrough = cv2.VideoCapture(STREAM_DEVICE_INDEX)
        
        if not cap_passthrough.isOpened():
            print(f"Error: Could not open video device index {STREAM_DEVICE_INDEX}. Check if OBS Virtual Camera is running and selected.")
            cap_passthrough = None 
            
        else:
            # Setting properties may help ensure consistency
            cap_passthrough.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
            cap_passthrough.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
            print(f"Passthrough stream opened from device index: {STREAM_DEVICE_INDEX}")

    except Exception as e:
        print(f"Error setting up video capture: {e}")
        return

    # --- Create visualization window ---
    cv2.namedWindow("Main Display", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Main Display", CAP_WIDTH, CAP_HEIGHT)

    # Variables for Ad Content Handling
    ad_cap = None
    
    try:
        while True:
            frame_passthrough = None
            ret_passthrough = False
            
            if cap_passthrough is not None:
                # 1. Capture the latest frame from the passthrough source
                ret_passthrough, frame_passthrough = cap_passthrough.read()
            
            # Create a black placeholder if the stream is unavailable or failed
            if not ret_passthrough or cap_passthrough is None:
                current_content = np.zeros((CAP_HEIGHT, CAP_WIDTH, 3), dtype=np.uint8)
                cv2.putText(current_content, "STREAM OFFLINE", 
                            (50, CAP_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 0, 255), 3)
            else:
                # Default content is the stream frame
                current_content = frame_passthrough
            
            # 2. Check for new prediction score
            score = 0.0
            if len(prediction_queue) > 0:
                score = prediction_queue[-1]
            
            # 3. Decision Logic (Ad vs. Passthrough)
            if score > 0.85 and content_files: 
                # --- Ad detected → Display Ad Content ---
                
                # Logic for picking/starting new ad remains the same
                if ad_cap is None:
                    content_file = random.choice(content_files)
                    
                    if content_file.lower().endswith((".png", ".jpg", ".jpeg")):
                        img = cv2.imread(content_file)
                        if img is not None:
                             current_content = cv2.resize(img, (CAP_WIDTH, CAP_HEIGHT))
                    else:
                        ad_cap = cv2.VideoCapture(content_file)
                        print(f"Starting video ad: {content_file}")

                if ad_cap is not None:
                    # Video Ad Playback
                    ret_ad, frame_ad = ad_cap.read()
                    if ret_ad:
                        current_content = cv2.resize(frame_ad, (CAP_WIDTH, CAP_HEIGHT))
                    else:
                        # Video Ad finished
                        ad_cap.release()
                        ad_cap = None
                        print("Video ad finished.")
                
            else:
                # --- Content detected → Display Passthrough Stream ---
                # current_content is already set to frame_passthrough (or black screen if offline)
                
                # Cleanup video ad if score dropped while it was playing
                if ad_cap is not None:
                    ad_cap.release()
                    ad_cap = None
                
                # Add text overlay if stream is live
                if cap_passthrough is not None and ret_passthrough:
                    cv2.putText(current_content, f"SPORTS GAME ({score:.1%})",
                                (50, CAP_HEIGHT - 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (255, 255, 255), 2)


            # 4. Display the Frame
            if current_content is not None:
                cv2.imshow("Main Display", current_content)

            # 5. Handle Exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nShutting down...")

    finally:
        # Release all resources
        if cap_passthrough is not None:
            cap_passthrough.release()
        if ad_cap is not None:
            ad_cap.release()
            
        cv2.destroyAllWindows()
        print("Cleanup complete.")

if __name__ == "__main__":
    main()