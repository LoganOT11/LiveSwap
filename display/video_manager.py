import cv2
import os
import random
import numpy as np

class VideoManager:
    def __init__(self, content_folder, stream_index=0):
        # Case-insensitive extension check
        self.files = [os.path.join(content_folder, f) for f in os.listdir(content_folder) 
                      if f.lower().endswith(('.png','.jpg','.jpeg','.mp4','.avi'))]
        
        self.cap = cv2.VideoCapture(stream_index)
        # Set hardware resolution (zero CPU cost)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        
        self.ad_cap, self.ad_img = None, None

    def get_frame(self, is_ad):
        # --- AD MODE ---
        if is_ad and self.files:
            # Initialize new content if nothing is playing
            if not self.ad_cap and self.ad_img is None:
                f = random.choice(self.files)
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img = cv2.imread(f)
                    # Store raw image without resizing
                    if img is not None: self.ad_img = img 
                else:
                    self.ad_cap = cv2.VideoCapture(f)

            # Render Image
            if self.ad_img is not None: 
                # CRITICAL FIX: Return a copy so cv2.putText doesn't overwrite the original in memory
                return self.ad_img.copy()
            
            # Render Video
            if self.ad_cap:
                ret, frame = self.ad_cap.read()
                if ret: 
                    return frame # No resize (fastest)
                
                # Video finished: Attempt to loop once
                self.ad_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.ad_cap.read()
                if ret: 
                    return frame
                
                # If loop fails, release resource
                self.ad_cap.release()
                self.ad_cap = None
        
        # --- PASSTHROUGH MODE ---
        if self.ad_cap: self.ad_cap.release(); self.ad_cap = None
        self.ad_img = None
        
        ret, frame = self.cap.read()
        # Fallback black frame if stream fails
        return frame if ret else np.zeros((600, 800, 3), np.uint8)

    def release(self):
        self.cap.release()
        if self.ad_cap: self.ad_cap.release()
