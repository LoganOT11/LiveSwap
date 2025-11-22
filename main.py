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
        print(f"No content files found in {CONTENT_FOLDER}")
        return

    # --- Create visualization window ---
    cv2.namedWindow("Main Display", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Main Display", 800, 600)

    try:
        current_content = None

        while True:
            if len(prediction_queue) > 0:
                score = prediction_queue[-1]  # newest value
                # print(f"[WINDOW] score={score:.3f}")

                if score > 0.85:
                    # Ad detected → display content
                    content_file = random.choice(content_files)
                    if content_file.lower().endswith((".png", ".jpg", ".jpeg")):
                        img = cv2.imread(content_file)
                        if img is None:
                            continue
                        current_content = cv2.resize(img, (800, 600))
                    else:
                        # For video files, open VideoCapture
                        cap = cv2.VideoCapture(content_file)
                        ret, frame = cap.read()
                        if ret:
                            current_content = cv2.resize(frame, (800, 600))
                        cap.release()
                else:
                    # Content detected → blank or green screen
                    current_content = 255 * np.ones((600, 800, 3), dtype=np.uint8)
                    cv2.putText(current_content, f"CONTENT ({score:.1%})",
                                (50, 300), cv2.FONT_HERSHEY_SIMPLEX,
                                2.0, (0, 0, 0), 4)

                cv2.imshow("Main Display", current_content)

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
