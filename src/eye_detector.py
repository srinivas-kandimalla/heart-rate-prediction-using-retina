import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None

class EyeDetector:
    def __init__(self):
        self.use_mediapipe = (mp is not None)
        self.face_mesh = None
        
        if self.use_mediapipe:
            try:
                self.mp_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                
                # Standard MediaPipe Face Mesh indices for Left and Right eye contours
                self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144, 163, 7, 145, 154, 155]
                self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380, 381, 382, 388, 386, 249]
            except Exception as e:
                print(f"[Detector Warning] MediaPipe Face Mesh init failed: {e}. Using geometric eyes.")
                self.use_mediapipe = False
                
        self.prev_left = None
        self.prev_right = None
                
    def detect_eyes(self, frame, face_bbox):
        """
        Detects left and right eye bounding boxes inside the frame.
        
        Returns:
            dict: {
                "left_eye": (lx, ly, lw, lh) or None,
                "right_eye": (rx, ry, rw, rh) or None
            }
        """
        if face_bbox is None:
            self.prev_left = None
            self.prev_right = None
            return {"left_eye": None, "right_eye": None}
            
        fx, fy, fw, fh = face_bbox
        h, w, _ = frame.shape
        
        left_eye_box = None
        right_eye_box = None
        mesh_success = False
        
        # 1. Try MediaPipe Face Mesh
        if self.use_mediapipe and self.face_mesh is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                results = self.face_mesh.process(frame_rgb)
                
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0].landmark
                    
                    # Helper function to get pixel box from landmark indices
                    def get_eye_bbox(indices):
                        pts = []
                        for idx in indices:
                            lm = face_landmarks[idx]
                            pts.append((int(lm.x * w), int(lm.y * h)))
                        pts = np.array(pts)
                        x, y, ew, eh = cv2.boundingRect(pts)
                        
                        # Add a small padding margin around the eye contour
                        pad_x = int(ew * 0.15)
                        pad_y = int(eh * 0.15)
                        x = max(0, x - pad_x)
                        y = max(0, y - pad_y)
                        ew = min(w - x, ew + 2 * pad_x)
                        eh = min(h - y, eh + 2 * pad_y)
                        return (x, y, ew, eh)
                        
                    left_eye_box = get_eye_bbox(self.LEFT_EYE_INDICES)
                    right_eye_box = get_eye_bbox(self.RIGHT_EYE_INDICES)
                    mesh_success = True
            except Exception as e:
                print(f"[Detector Warning] Face Mesh processing failed: {e}")
                
        # 2. Geometric Fallback based on Face Bounding Box
        if not mesh_success:
            # Viewer's left eye (Patient's right eye)
            cx1 = fx + int(fw * 0.33)
            cy1 = fy + int(fh * 0.38)
            
            # Viewer's right eye (Patient's left eye)
            cx2 = fx + int(fw * 0.67)
            cy2 = fy + int(fh * 0.38)
            
            # Eye box half-width (radius)
            er = int(fw * 0.09)
            
            # Left eye (viewer's left)
            lx = max(0, cx1 - er)
            ly = max(0, cy1 - er)
            lw = min(w - lx, er * 2)
            lh = min(h - ly, er * 2)
            left_eye_box = (lx, ly, lw, lh)
            
            # Right eye (viewer's right)
            rx = max(0, cx2 - er)
            ry = max(0, cy2 - er)
            rw = min(w - rx, er * 2)
            rh = min(h - ry, er * 2)
            right_eye_box = (rx, ry, rw, rh)
            
        # 3. Apply smoothing to boxes to prevent jitter
        alpha = 0.25
        if left_eye_box:
            if self.prev_left is None:
                self.prev_left = left_eye_box
            else:
                lx, ly, lw, lh = left_eye_box
                lx_p, ly_p, lw_p, lh_p = self.prev_left
                lx = int(lx_p + alpha * (lx - lx_p))
                ly = int(ly_p + alpha * (ly - ly_p))
                lw = int(lw_p + alpha * (lw - lw_p))
                lh = int(lh_p + alpha * (lh - lh_p))
                left_eye_box = (lx, ly, lw, lh)
                self.prev_left = left_eye_box
        else:
            self.prev_left = None

        if right_eye_box:
            if self.prev_right is None:
                self.prev_right = right_eye_box
            else:
                rx, ry, rw, rh = right_eye_box
                rx_p, ry_p, rw_p, rh_p = self.prev_right
                rx = int(rx_p + alpha * (rx - rx_p))
                ry = int(ry_p + alpha * (ry - ry_p))
                rw = int(rw_p + alpha * (rw - rw_p))
                rh = int(rh_p + alpha * (rh - rh_p))
                right_eye_box = (rx, ry, rw, rh)
                self.prev_right = right_eye_box
        else:
            self.prev_right = None
            
        return {"left_eye": left_eye_box, "right_eye": right_eye_box}
        
    def draw_eye_hud(self, frame, eye_boxes):
        """
        Draws green (Left eye) and orange (Right eye) circles directly on the frame.
        """
        # 1. Left Eye Circle (Viewer's Left, Patient's Right - Orange)
        left_box = eye_boxes.get("left_eye")
        if left_box:
            lx, ly, lw, lh = left_box
            cx = lx + lw // 2
            cy = ly + lh // 2
            r = min(lw, lh) // 2
            cv2.circle(frame, (cx, cy), r, (0, 180, 240), 2)
            
        # 2. Right Eye Circle (Viewer's Right, Patient's Left - Green)
        right_box = eye_boxes.get("right_eye")
        if right_box:
            rx, ry, rw, rh = right_box
            cx = rx + rw // 2
            cy = ry + rh // 2
            r = min(rw, rh) // 2
            cv2.circle(frame, (cx, cy), r, (0, 255, 0), 2)
            
    def crop_eye_rois(self, frame, eye_boxes):
        """
        Crops left and right eye regions from BGR frame.
        
        Returns:
            dict: {"left_roi": ndarray or None, "right_roi": ndarray or None}
        """
        left_box = eye_boxes.get("left_eye")
        right_box = eye_boxes.get("right_eye")
        
        left_roi = None
        right_roi = None
        
        if left_box:
            lx, ly, lw, lh = left_box
            if lw > 0 and lh > 0:
                left_roi = frame[ly:ly+lh, lx:lx+lw].copy()
                
        if right_box:
            rx, ry, rw, rh = right_box
            if rw > 0 and rh > 0:
                right_roi = frame[ry:ry+rh, rx:rx+rw].copy()
                
        return {"left_roi": left_roi, "right_roi": right_roi}
