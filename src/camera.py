import cv2
import numpy as np
import time
from PyQt5.QtCore import QThread, pyqtSignal

class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    
    def __init__(self, camera_id=0, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.running = False
        self.demo_mode = False
        
    def run(self):
        self.running = True
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            print("[Camera Warning] Physical webcam not available. Switching to Simulated Face Demo Mode.")
            self.demo_mode = True
            
        start_time = time.time()
        sim_fps = 30.0
        
        while self.running:
            frame_start = time.time()
            
            if self.demo_mode:
                # 1. Procedurally generate a realistic synthetic face frame
                frame = np.zeros((360, 480, 3), dtype=np.uint8)
                # Soft blue-dark medical grid background
                frame[:, :] = (20, 18, 15)
                
                # Draw grid lines for a technical look
                for y in range(0, 360, 40):
                    cv2.line(frame, (0, y), (480, y), (30, 26, 22), 1)
                for x in range(0, 480, 40):
                    cv2.line(frame, (x, 0), (x, 360), (30, 26, 22), 1)
                    
                # Draw Face Circle (skin tone BGR)
                face_center = (240, 180)
                cv2.circle(frame, face_center, 90, (145, 188, 225), -1)
                
                # Draw eyes (white ellipses)
                eye_l = (205, 170)
                eye_r = (275, 170)
                
                # Handle occasional simulated blinking
                blink = int(time.time() * 0.8) % 4 == 0
                if blink:
                    # Draw closed eyelids
                    cv2.line(frame, (eye_l[0]-15, eye_l[1]), (eye_l[0]+15, eye_l[1]), (80, 120, 160), 2)
                    cv2.line(frame, (eye_r[0]-15, eye_r[1]), (eye_r[0]+15, eye_r[1]), (80, 120, 160), 2)
                else:
                    # Draw open eyes
                    cv2.ellipse(frame, eye_l, (18, 10), 0, 0, 360, (255, 255, 255), -1)
                    cv2.ellipse(frame, eye_r, (18, 10), 0, 0, 360, (255, 255, 255), -1)
                    
                    # Pupil movement simulation (gaze drift)
                    drift_x = int(2 * np.sin(time.time() * 1.5))
                    cv2.circle(frame, (eye_l[0] + drift_x, eye_l[1]), 6, (40, 40, 40), -1)
                    cv2.circle(frame, (eye_r[0] + drift_x, eye_r[1]), 6, (40, 40, 40), -1)
                    
                # Draw mouth
                cv2.ellipse(frame, (240, 220), (22, 6), 0, 0, 180, (80, 85, 185), 2)
                
                # Draw a nose bridge line
                cv2.line(frame, (240, 175), (240, 205), (110, 150, 195), 2)
                
                # 2. Modulate face color green channel slightly to simulate cardiac blood volume changes
                # Set a target heart rate of 74.0 BPM (frequency = 1.233 Hz)
                target_bpm = 74.0
                freq = target_bpm / 60.0
                elapsed = time.time() - start_time
                
                # Modulate average color values inside the facial circle
                pulse = 3.5 * np.sin(2.0 * np.pi * freq * elapsed)
                pulse += np.random.normal(0, 0.1) # Add minor sensor noise
                
                # Overlay color modulation
                mask = np.zeros((360, 480), dtype=np.uint8)
                cv2.circle(mask, face_center, 90, 255, -1)
                
                # Add pulse offset only to the green channel of the face ROI
                face_green = frame[:, :, 1].astype(np.float32)
                face_green[mask > 0] += pulse
                frame[:, :, 1] = np.clip(face_green, 0, 255).astype(np.uint8)
                
                self.frame_ready.emit(frame)
                
                # Sleep to maintain 30 FPS frame rate
                elapsed_frame = time.time() - frame_start
                sleep_time = max(1.0 / sim_fps - elapsed_frame, 0)
                time.sleep(sleep_time)
                
            else:
                ret, frame = cap.read()
                if ret:
                    self.frame_ready.emit(frame)
                else:
                    time.sleep(0.01)
                    
        if cap.isOpened():
            cap.release()
            
    def stop(self):
        self.running = False
        self.wait()
