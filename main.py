import cv2
import time
import threading
from collections import deque
from prediction.predictor import predict_live
import settings
import numpy as np

def draw_status_image(score, window_name):
    """Helper function to create the visualization image based on score."""
    # Make blank display
    img = np.zeros((300, 500, 3), dtype=np.uint8)
    
    # If we haven't received a score yet
    if score is None:
        cv2.putText(img, "Waiting...", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return img

    if score > 0.85:
        # AD detected = red screen
        img[:, :] = [0, 0, 255]  # BGR
        text = f"AD ({score:.1%})"
    else:
        # CONTENT = green screen
        img[:, :] = [0, 255, 0]  # BGR
        text = f"CONTENT ({score:.1%})"

    # Draw the main text
    cv2.putText(img, text, (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)
    # Draw the window title on the image for clarity
    cv2.putText(img, window_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    return img

def main():
    # --- SETUP QUEUES ---
    # We need separate queues so the threads don't conflict
    queue_model_1 = deque(maxlen=1)
    queue_model_2 = deque(maxlen=1)

    # --- THREAD 1: Original Model ---
    thread_1 = threading.Thread(
        target=predict_live,
        kwargs={
            "model_path": settings.MODEL_PATH,
            "device_index": 2,  
            "show_hud": False,
            "prediction_queue": queue_model_1
        },
        daemon=True
    )
    thread_1.start()

    # --- THREAD 2: Second Model (MODEL_PATH2) ---
    thread_2 = threading.Thread(
        target=predict_live,
        kwargs={
            "model_path": settings.MODEL_PATH2, # Using the second model path
            "device_index": 2,  # Listening to the same audio device
            "show_hud": False,
            "prediction_queue": queue_model_2
        },
        daemon=True
    )
    thread_2.start()

    # --- SETUP WINDOWS ---
    cv2.namedWindow("Model 1", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Model 1", 500, 300)

    cv2.namedWindow("Model 2", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Model 2", 500, 300)
    
    # Move window 2 slightly so they don't overlap perfectly on launch
    cv2.moveWindow("Model 2", 520, 0) 

    # Current scores (init as None)
    score_1 = None
    score_2 = None

    try:
        while True:
            # 1. Check Queue 1
            if len(queue_model_1) > 0:
                score_1 = queue_model_1[-1]

            # 2. Check Queue 2
            if len(queue_model_2) > 0:
                score_2 = queue_model_2[-1]

            # 3. Generate Images
            img_1 = draw_status_image(score_1, "Model 1")
            img_2 = draw_status_image(score_2, "Model 2")

            # 4. Update Displays
            cv2.imshow("Model 1", img_1)
            cv2.imshow("Model 2", img_2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.05)  # smooth UI loop

    except KeyboardInterrupt:
        print("\nShutting down...")

    finally:
        cv2.destroyAllWindows()
        print("Cleanup complete.")

if __name__ == "__main__":
    main()
