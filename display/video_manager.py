import cv2
import os
import random
import numpy as np

class VideoManager:
    def __init__(self, content_folder, stream_index=0):
        self.files = [os.path.join(content_folder, f) for f in os.listdir(content_folder) if f.lower().endswith(('.png','.jpg','.mp4','.avi'))]
        self.cap = cv2.VideoCapture(stream_index)
        # Set hardware resolution once to avoid CPU-heavy resizing every loop
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        self.ad_cap, self.ad_img = None, None

    def get_frame(self, is_ad):
        # --- AD MODE ---
        if is_ad and self.files:
            if not self.ad_cap and not self.ad_img: # Load new content if none active
                f = random.choice(self.files)
                if f.endswith(('.png', '.jpg')): 
                    img = cv2.imread(f)
                    if img is not None: self.ad_img = cv2.resize(img, (800, 600))
                else: 
                    self.ad_cap = cv2.VideoCapture(f)
            
            if self.ad_img is not None: return self.ad_img
            if self.ad_cap:
                ret, frame = self.ad_cap.read()
                if ret: return cv2.resize(frame, (800, 600))
                self.ad_cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video automatically
                return self.get_frame(is_ad) # Retry frame 0
        
        # --- PASSTHROUGH MODE ---
        if self.ad_cap: self.ad_cap.release(); self.ad_cap = None
        self.ad_img = None
        
        ret, frame = self.cap.read()
        return frame if ret else np.zeros((600, 800, 3), np.uint8)

    def release(self):
        self.cap.release()
        if self.ad_cap: self.ad_cap.release()
