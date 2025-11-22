import cv2
import numpy as np

# Constants for the Dashboard Layout
WINDOW_NAME = "Ad Detector Dashboard"
ROW_HEIGHT = 120
ROW_WIDTH = 600

def draw_status_bar(score, label, max_val=1.0, threshold=0.85):
    """
    Draws a single horizontal status bar.
    """
    img = np.zeros((ROW_HEIGHT, ROW_WIDTH, 3), dtype=np.uint8)
    
    if score is None:
        cv2.putText(img, "Buffering...", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.rectangle(img, (0, 0), (ROW_WIDTH-1, ROW_HEIGHT-1), (50, 50, 50), 2)
        return img

    # Logic
    is_ad = score > threshold
    color = [0, 0, 255] if is_ad else [0, 255, 0]  # Red vs Green
    text_status = "ADVERTISEMENT" if is_ad else "CONTENT"
    
    # Draw Filled Bar
    fill_ratio = min(score / max_val, 1.0)
    fill_width = int(fill_ratio * ROW_WIDTH)
    
    # Background Bar
    cv2.rectangle(img, (0, 0), (fill_width, ROW_HEIGHT), color, -1)
    
    # Text
    cv2.putText(img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(img, text_status, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    
    score_text = f"{score:.2f} / {max_val} (Thresh: {threshold})"
    cv2.putText(img, score_text, (ROW_WIDTH - 280, ROW_HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Border
    cv2.rectangle(img, (0, 0), (ROW_WIDTH-1, ROW_HEIGHT-1), (50, 50, 50), 2)
    
    return img

def init_dashboard():
    """Creates the window."""
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    # 5 Rows * 120px = 600px height
    cv2.resizeWindow(WINDOW_NAME, ROW_WIDTH, ROW_HEIGHT * 5)

def update_dashboard(scores):
    """
    Calculates sums, generates images, stacks them, and handles the window.
    Returns True if user pressed 'q' to quit, False otherwise.
    """
    # Unpack scores (handle Nones safely)
    s_fast = scores.get('fast') or 0.0
    s_slow = scores.get('slow') or 0.0
    s_exp  = scores.get('exp')  or 0.0

    # Calculate Sums
    sum_2 = s_fast + s_slow
    sum_3 = s_fast + s_slow + s_exp

    # Generate 5 Panels
    panels = [
        draw_status_bar(s_fast, "1. Fast Sentry (1.0s)", max_val=1.0, threshold=0.85),
        draw_status_bar(s_slow, "2. Main Judge (4.3s)",  max_val=1.0, threshold=0.85),
        draw_status_bar(s_exp,  "3. Experiment (1.0s)",  max_val=1.0, threshold=0.85),
        draw_status_bar(sum_2,  "4. Sum (Fast + Slow)",  max_val=2.0, threshold=0.70),
        draw_status_bar(sum_3,  "5. Sum (All Three)",    max_val=3.0, threshold=1.00)
    ]

    # Stack
    dashboard = np.vstack(panels)

    # Show
    cv2.imshow(WINDOW_NAME, dashboard)

    # Check Input (Return True if 'q' is pressed)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return True
    
    return False

def close_dashboard():
    cv2.destroyAllWindows()
