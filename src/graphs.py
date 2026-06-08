import pyqtgraph as pg
from PyQt5.QtCore import Qt
import numpy as np


class PulseGraph(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Card background #0C1424
        self.setBackground((12, 20, 36))
        self.showGrid(x=True, y=True, alpha=0.15)
        self.setMouseEnabled(x=False, y=False)
        self.enableAutoRange(axis="y", enable=True)
        self.enableAutoRange(axis="x", enable=True)

        self.getAxis("left").setPen((35, 50, 80))
        self.getAxis("bottom").setPen((35, 50, 80))

        # Set axes labels
        self.setLabel("left", "Amplitude", color="#8F9CA3", size="7.5pt")
        self.setLabel("bottom", "Time (s)", color="#8F9CA3", size="7.5pt")

        # Neon Green Pen for the time domain signal
        self.pen = pg.mkPen(color=(57, 255, 20), width=2.0)
        self.curve = self.plot(pen=self.pen, antialias=True)

    def update_signal(self, sig_data):
        """
        Updates the scrolling plot with processed signal array.
        """
        if len(sig_data) > 0:
            self.curve.setData(sig_data)
        else:
            self.curve.clear()


class SpectrumGraph(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Card background #0C1424
        self.setBackground((12, 20, 36))
        self.showGrid(x=True, y=True, alpha=0.15)
        self.setMouseEnabled(x=False, y=False)
        self.enableAutoRange(axis="y", enable=True)
        self.enableAutoRange(axis="x", enable=True)

        self.getAxis("left").setPen((35, 50, 80))
        self.getAxis("bottom").setPen((35, 50, 80))

        # Set axes labels
        self.setLabel("left", "Power", color="#8F9CA3", size="7.5pt")
        self.setLabel("bottom", "Frequency (Hz)", color="#8F9CA3", size="7.5pt")

        # Cyber-cyan Pen for the FFT curve
        self.pen = pg.mkPen(color=(0, 255, 255), width=2.0)
        self.curve = self.plot(pen=self.pen, antialias=True)

        # Vertical peak frequency red marker line
        self.peak_line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(color=(255, 59, 48), width=1.5, style=Qt.DashLine),
        )
        self.addItem(self.peak_line)
        self.peak_line.hide()

    def update_spectrum(self, freqs, fft_vals, peak_freq=None):
        """
        Updates the frequency graph and positions the red indicator.
        """
        if len(freqs) > 0 and len(fft_vals) > 0:
            # Only display frequencies up to 4.0 Hz (240 BPM)
            cut_idx = np.where(freqs <= 4.0)[0]
            if len(cut_idx) > 0:
                show_freqs = freqs[cut_idx]
                show_vals = fft_vals[cut_idx]

                self.curve.setData(show_freqs, show_vals)

                # Position peak marker line
                if peak_freq is not None and peak_freq > 0:
                    self.peak_line.setValue(peak_freq)
                    self.peak_line.show()
                else:
                    self.peak_line.hide()
                return

        self.curve.clear()
        self.peak_line.hide()
