import numpy as np

class RetinaAnalyzer:
    def __init__(self):
        pass
        
    def extract_green_intensity(self, eye_rois):
        """
        Extracts the average green channel intensity from the left and right eye regions.
        In BGR format, index 1 corresponds to the Green channel.
        
        Args:
            eye_rois (dict): Dictionary containing BGR crop images {"left_roi": ndarray, "right_roi": ndarray}
            
        Returns:
            float: Spatial average green channel intensity.
        """
        left_roi = eye_rois.get("left_roi")
        right_roi = eye_rois.get("right_roi")
        
        intensities = []
        
        # Calculate mean green channel value for Left Eye
        if left_roi is not None and left_roi.size > 0:
            intensities.append(np.mean(left_roi[:, :, 1]))
            
        # Calculate mean green channel value for Right Eye
        if right_roi is not None and right_roi.size > 0:
            intensities.append(np.mean(right_roi[:, :, 1]))
            
        if len(intensities) == 0:
            return 0.0
            
        # Return average of both eye regions
        return float(np.mean(intensities))
