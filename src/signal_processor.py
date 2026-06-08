import numpy as np
import time

try:
    from scipy.signal import butter, filtfilt
except ImportError:
    butter = None
    filtfilt = None

class SignalProcessor:
    def __init__(self, buffer_size=150):
        self.buffer_size = buffer_size
        self.signal_buffer = []
        self.timestamps = []
        
    def add_value(self, val):
        """
        Adds a new intensity reading and timestamp to the rolling buffers.
        """
        self.signal_buffer.append(val)
        self.timestamps.append(time.time())
        
        # Maintain window size
        if len(self.signal_buffer) > self.buffer_size:
            self.signal_buffer.pop(0)
            self.timestamps.pop(0)
            
    def get_fps(self):
        """
        Calculates the dynamic frame rate based on buffer timestamps.
        """
        if len(self.timestamps) < 10:
            return 30.0  # default assumption
            
        time_elapsed = self.timestamps[-1] - self.timestamps[0]
        if time_elapsed <= 0:
            return 30.0
            
        return len(self.timestamps) / time_elapsed
        
    def process(self):
        """
        Detrends and bandpass-filters the raw rolling signal.
        
        Returns:
            numpy.ndarray: Processed pulse waveform.
        """
        n = len(self.signal_buffer)
        if n < 30:
            return np.array([])
            
        sig = np.array(self.signal_buffer, dtype=np.float32)
        
        # 1. Linear detrend (remove DC baseline drift)
        sig = sig - np.mean(sig)
        
        # 2. Calculate dynamic FPS
        fps = self.get_fps()
        nyq = 0.5 * fps
        
        # Bandpass parameters: 45 BPM (0.75 Hz) to 180 BPM (3.0 Hz)
        lowcut = 0.75
        highcut = 3.0
        
        # 3. Apply Butterworth Bandpass Filter using SciPy
        if butter is not None and filtfilt is not None and nyq > highcut:
            try:
                low = lowcut / nyq
                high = highcut / nyq
                b, a = butter(2, [low, high], btype='band')
                filtered = filtfilt(b, a, sig)
                return filtered
            except Exception as e:
                # Fallback on calculation failure
                pass
                
        # 4. Fallback filter (Moving Average convolution if SciPy is missing or fails)
        window_size = 5
        smoothed = np.convolve(sig, np.ones(window_size)/window_size, mode='same')
        return smoothed
