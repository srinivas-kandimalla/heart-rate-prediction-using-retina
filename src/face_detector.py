import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None

class FaceDetector:
    def __init__(self):
        # 1. Initialize MediaPipe Face Detection if available
        self.use_mediapipe = (mp is not None)
        self.face_detection = None
        self.cascade_detector = None
        
        if self.use_mediapipe:
            try:
                self.mp_face = mp.solutions.face_detection
                self.face_detection = self.mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
            except Exception as e:
                print(f"[Detector Warning] MediaPipe initialization failed: {e}. Falling back to Haar Cascades.")
                self.use_mediapipe = False
                
        if not self.use_mediapipe:
            # Fallback to standard OpenCV Haar Cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.cascade_detector = cv2.CascadeClassifier(cascade_path)
            
        self.prev_bbox = None
            
    def detect(self, frame):
        """
        Detects the primary face in the frame.
        
        Returns:
          tuple: (face_box, confidence, success)
                 face_box is (x, y, w, h)
                 confidence is float [0.0, 1.0]
        """
        h, w, _ = frame.shape
        
        if self.use_mediapipe and self.face_detection is not None:
            # Convert frame to RGB for MediaPipe processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                results = self.face_detection.process(frame_rgb)
                
                if results.detections:
                    # Select the largest detected face
                    largest_det = max(results.detections, key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height)
                    bbox = largest_det.location_data.relative_bounding_box
                    conf = largest_det.score[0]
                    
                    # Convert relative coordinates to pixels
                    fx = int(bbox.xmin * w)
                    fy = int(bbox.ymin * h)
                    fw = int(bbox.width * w)
                    fh = int(bbox.height * h)
                    
                    # Bound checking
                    fx = max(0, fx)
                    fy = max(0, fy)
                    fw = min(w - fx, fw)
                    fh = min(h - fy, fh)
                    
                    # Exponential smoothing for smooth visual tracking
                    bbox_raw = (fx, fy, fw, fh)
                    if self.prev_bbox is None:
                        self.prev_bbox = bbox_raw
                    else:
                        alpha = 0.25
                        x_p, y_p, w_p, h_p = self.prev_bbox
                        fx = int(x_p + alpha * (fx - x_p))
                        fy = int(y_p + alpha * (fy - y_p))
                        fw = int(w_p + alpha * (fw - w_p))
                        fh = int(h_p + alpha * (fh - h_p))
                        self.prev_bbox = (fx, fy, fw, fh)
                    
                    return (fx, fy, fw, fh), conf, True
            except Exception as e:
                print(f"[Detector Warning] MediaPipe detection failed: {e}. Switching to Cascade.")
                
        # Cascade Fallback
        if self.cascade_detector is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.cascade_detector.detectMultiScale(gray, 1.15, 5, minSize=(100, 100))
            if len(faces) > 0:
                # Sort by area
                faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                fx, fy, fw, fh = faces[0]
                
                # Exponential smoothing
                bbox_raw = (fx, fy, fw, fh)
                if self.prev_bbox is None:
                    self.prev_bbox = bbox_raw
                else:
                    alpha = 0.25
                    x_p, y_p, w_p, h_p = self.prev_bbox
                    fx = int(x_p + alpha * (fx - x_p))
                    fy = int(y_p + alpha * (fy - y_p))
                    fw = int(w_p + alpha * (fw - w_p))
                    fh = int(h_p + alpha * (fh - h_p))
                    self.prev_bbox = (fx, fy, fw, fh)
                    
                return (fx, fy, fw, fh), 0.85, True
                
        self.prev_bbox = None
        return None, 0.0, False
        
    def draw_hud_bracket(self, frame, bbox, confidence):
        """
        Draws professional hospital-system HUD brackets around the face.
        """
        if bbox is None:
            return
            
        fx, fy, fw, fh = bbox
        
        # Cyber Cyan for HUD target (B=255, G=255, R=0 in BGR)
        color = (255, 255, 0) if confidence > 0.6 else (0, 159, 255) # Cyan for high, Orange for low
        thickness = 2
        length = min(20, int(fw * 0.15))
        
        # Draw tech corner brackets
        # Top-Left
        cv2.line(frame, (fx, fy), (fx + length, fy), color, thickness)
        cv2.line(frame, (fx, fy), (fx, fy + length), color, thickness)
        # Top-Right
        cv2.line(frame, (fx + fw, fy), (fx + fw - length, fy), color, thickness)
        cv2.line(frame, (fx + fw, fy), (fx + fw, fy + length), color, thickness)
        # Bottom-Left
        cv2.line(frame, (fx, fy + fh), (fx + length, fy + fh), color, thickness)
        cv2.line(frame, (fx, fy + fh), (fx, fy + fh - length), color, thickness)
        # Bottom-Right
        cv2.line(frame, (fx + fw, fy + fh), (fx + fw - length, fy + fh), color, thickness)
        cv2.line(frame, (fx + fw, fy + fh), (fx + fw, fy + fh - length), color, thickness)
        
        # Draw a thin boundary box for target tracking feel
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), color, 1)
        
        # Center Target Reticle
        cx = fx + fw // 2
        cy = fy + fh // 2
        cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 8, 1)
        
        # Print Tracking Info Label
        cv2.putText(frame, "FACE LOCKED", (fx + 6, fy + 18), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (57, 255, 20), 1, cv2.LINE_AA) # Success green lock text
        cv2.putText(frame, f"TRK ACC: {confidence*100:.1f}%", (fx + 6, fy + fh - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
