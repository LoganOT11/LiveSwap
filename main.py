import cv2
import time
import threading
from collections import deque
from prediction.predictor import predict_live
import settings
import numpy as np

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

    # --- Create a simple visualization window ---
    cv2.namedWindow("Main Display", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Main Display", 500, 300)

    try:
        while True:
            if len(prediction_queue) > 0:
                score = prediction_queue[-1]
                # print(f"[WINDOW] score={score:.3f}")

                # Make blank display
                img = np.zeros((300, 500, 3), dtype=np.uint8)
                
                if score > 0.85:
                    # AD detected = red screen
                    img[:, :] = [0, 0, 255]  # BGR
                    cv2.putText(img, f"AD ({score:.1%})",
                                (30,150), cv2.FONT_HERSHEY_SIMPLEX,
                                1.5, (0,0,0), 4)
                else:
                    # CONTENT = green screen
                    img[:, :] = [0, 255, 0]  # BGR
                    cv2.putText(img, f"CONTENT ({score:.1%})",
                                (30,150), cv2.FONT_HERSHEY_SIMPLEX,
                                1.5, (0,0,0), 4)

                cv2.imshow("Main Display", img)

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
