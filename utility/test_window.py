import cv2
import numpy as np

print("Attempting to open window...")

# 1. Create a simple black square (Height, Width, Channels)
# We use a black image because "no content" (None) will crash OpenCV.
# A black image IS "no content" in computer vision terms.
test_image = np.zeros((300, 300, 3), dtype=np.uint8)

# 2. Open the Window
cv2.imshow("TEST WINDOW - Press any key to close", test_image)

# 3. CRITICAL: You must wait.
# Without waitKey, the script finishes instantly and the window closes 
# before your monitor can even draw it.
cv2.waitKey(0) 

cv2.destroyAllWindows()
print("Closed.")
