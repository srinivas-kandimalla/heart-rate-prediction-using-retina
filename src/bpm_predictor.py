import numpy as np

class BPMPredictor:
    def __init__(self):
        pass
        
    def predict(self, processed_signal, fps):
        """
        Executes FFT frequency analysis on the filtered signal to predict BPM.
        
        Returns:
            dict: Heart rate metrics and classification status.
        """
        n = len(processed_signal)
        if n < 50 or fps <= 0:
            return {
                "bpm": 0.0,
                "hz": 0.0,
                "confidence": 0.0,
                "status": "CALIBRATING...",
                "alert_level": "info",
                "freqs": np.array([]),
                "fft_vals": np.array([])
            }
            
        # 1. Compute Fast Fourier Transform
        fft_vals = np.abs(np.fft.rfft(processed_signal))
        fft_freqs = np.fft.rfftfreq(n, d=1.0/fps)
        
        # Focus on physiological heart rate bounds: 0.75 Hz to 2.5 Hz (45 to 150 BPM)
        valid_idx = np.where((fft_freqs >= 0.75) & (fft_freqs <= 2.50))[0]
        
        if len(valid_idx) > 0:
            # Locate the peak frequency index
            peak_idx = valid_idx[np.argmax(fft_vals[valid_idx])]
            peak_freq = fft_freqs[peak_idx]
            
            # Parabolic interpolation to find the true peak frequency
            if 0 < peak_idx < len(fft_vals) - 1:
                y1 = fft_vals[peak_idx - 1]
                y2 = fft_vals[peak_idx]
                y3 = fft_vals[peak_idx + 1]
                denom = y1 - 2 * y2 + y3
                if abs(denom) > 1e-5:
                    d = 0.5 * (y1 - y3) / denom
                    d = np.clip(d, -0.5, 0.5)
                    peak_freq = fft_freqs[peak_idx] + d * (fft_freqs[1] - fft_freqs[0])
            
            bpm = peak_freq * 60.0
            hz = peak_freq
            
            # 2. Calculate confidence based on Signal-to-Noise Ratio (SNR) in the band
            peak_amplitude = fft_vals[peak_idx]
            other_indices = [idx for idx in valid_idx if idx != peak_idx]
            if len(other_indices) > 0:
                mean_noise = np.mean(fft_vals[other_indices])
                if mean_noise > 0:
                    snr = peak_amplitude / mean_noise
                    # Map SNR ratio to a percentage [10.0% to 99.8%]
                    # SNR of 1.0 -> 20.0%, SNR of 3.0 -> 80.0%, SNR >= 3.66 -> 99.8%
                    confidence = float(np.clip((snr - 1.0) * 30.0 + 20.0, 10.0, 99.8))
                else:
                    confidence = 99.8
            else:
                confidence = 50.0
        else:
            bpm = 0.0
            hz = 0.0
            confidence = 0.0
            
        # 3. Classify cardiovascular health status
        status = "NORMAL"
        alert_level = "success"  # green
        
        if bpm > 0:
            if bpm < 60.0:
                status = "BRADYCARDIA"
                alert_level = "warning"  # orange
            elif 60.0 <= bpm <= 100.0:
                status = "NORMAL"
                alert_level = "success"  # green
            elif 100.0 < bpm <= 120.0:
                status = "ELEVATED"
                alert_level = "caution"  # yellow
            else:
                status = "TACHYCARDIA"
                alert_level = "danger"   # red
        else:
            status = "CALIBRATING..."
            alert_level = "info"
            
        return {
            "bpm": float(bpm),
            "hz": float(hz),
            "confidence": float(confidence),
            "status": status,
            "alert_level": alert_level,
            "freqs": fft_freqs,
            "fft_vals": fft_vals
        }
