import cv2
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QFrame,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap, QImage
from src.graphs import PulseGraph, SpectrumGraph


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(
            "Heart Rate Prediction Using Retina - Real-Time Monitor Dashboard"
        )
        self.setMinimumSize(1150, 720)
        self.resize(1180, 750)

        # Core Stylesheet matching the target UI design
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #080E1C;
            }
            QFrame#card {
                background-color: #0C1424;
                border-radius: 6px;
            }
            QLabel {
                color: #FFFFFF;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
        """
        )

        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 12, 15, 12)
        main_layout.setSpacing(12)

        # ==========================================
        # HEADER PANEL (Title & Monitoring Status)
        # ==========================================
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(5, 5, 5, 2)

        title_vbox = QVBoxLayout()
        title_vbox.setSpacing(1)

        lbl_title = QLabel("HEART RATE PREDICTION USING RETINA")
        lbl_title.setStyleSheet(
            "font-size: 13pt; color: #FFFFFF; font-weight: bold; letter-spacing: 0.5px;"
        )

        lbl_subtitle = QLabel("Real-Time Retinal Cardiovascular Monitoring System")
        lbl_subtitle.setStyleSheet(
            "font-size: 8.5pt; color: #00B4D8; font-weight: bold;"
        )

        title_vbox.addWidget(lbl_title)
        title_vbox.addWidget(lbl_subtitle)
        header_layout.addLayout(title_vbox)

        header_layout.addStretch()

        # Monitoring Status
        self.lbl_monitoring_active = QLabel("● MONITORING ACTIVE")
        self.lbl_monitoring_active.setStyleSheet(
            "font-size: 8.5pt; color: #39FF14; font-weight: bold;"
        )
        header_layout.addWidget(self.lbl_monitoring_active, 0, Qt.AlignVCenter)

        main_layout.addLayout(header_layout)

        # ==========================================
        # BODY DASHBOARD (Split Left & Right)
        # ==========================================
        body_layout = QHBoxLayout()
        body_layout.setSpacing(12)

        # ------------------------------------------
        # LEFT COLUMN: CAMERA & EYE REGION ROI
        # ------------------------------------------
        left_layout = QVBoxLayout()
        left_layout.setSpacing(12)

        # Camera Feed Card
        cam_card = QFrame()
        cam_card.setObjectName("card")
        cam_card_layout = QVBoxLayout(cam_card)
        cam_card_layout.setContentsMargins(6, 6, 6, 6)
        cam_card_layout.setAlignment(Qt.AlignCenter)

        self.camera_feed_label = QLabel("CAMERA BOOTING...")
        self.camera_feed_label.setAlignment(Qt.AlignCenter)
        self.camera_feed_label.setStyleSheet(
            "background-color: #050B18; border-radius: 4px; color: #2A3B5C; font-weight: bold;"
        )
        self.camera_feed_label.setFixedSize(480, 360)
        cam_card_layout.addWidget(self.camera_feed_label)

        left_layout.addWidget(cam_card)

        # Eye Region ROI Card
        roi_card = QFrame()
        roi_card.setObjectName("card")
        roi_card_layout = QVBoxLayout(roi_card)
        roi_card_layout.setContentsMargins(15, 12, 15, 15)
        roi_card_layout.setSpacing(10)

        lbl_roi_header = QLabel("EYE REGION (ROI)")
        lbl_roi_header.setStyleSheet(
            "font-size: 8.5pt; color: #FFFFFF; font-weight: bold; letter-spacing: 0.5px;"
        )
        roi_card_layout.addWidget(lbl_roi_header)

        roi_grid = QHBoxLayout()
        roi_grid.setSpacing(15)

        # Left Eye Column
        left_eye_vbox = QVBoxLayout()
        left_eye_vbox.setSpacing(6)
        lbl_left_title = QLabel("LEFT EYE")
        lbl_left_title.setStyleSheet(
            "font-size: 7.5pt; color: #00B4D8; font-weight: bold;"
        )
        lbl_left_title.setAlignment(Qt.AlignCenter)

        self.left_roi_label = QLabel("NO SIGNAL")
        self.left_roi_label.setAlignment(Qt.AlignCenter)
        self.left_roi_label.setFixedSize(210, 80)
        self.left_roi_label.setStyleSheet(
            "background-color: #050B18; border-radius: 4px; color: #5E6E88; font-size: 9px;"
        )

        self.lbl_l_sig = QLabel("● GOOD SIGNAL")
        self.lbl_l_sig.setStyleSheet(
            "color: #39FF14; font-size: 8pt; font-weight: bold;"
        )
        self.lbl_l_sig.setAlignment(Qt.AlignCenter)

        left_eye_vbox.addWidget(lbl_left_title)
        left_eye_vbox.addWidget(self.left_roi_label, 0, Qt.AlignCenter)
        left_eye_vbox.addWidget(self.lbl_l_sig)
        roi_grid.addLayout(left_eye_vbox)

        # Right Eye Column
        right_eye_vbox = QVBoxLayout()
        right_eye_vbox.setSpacing(6)
        lbl_right_title = QLabel("RIGHT EYE")
        lbl_right_title.setStyleSheet(
            "font-size: 7.5pt; color: #00B4D8; font-weight: bold;"
        )
        lbl_right_title.setAlignment(Qt.AlignCenter)

        self.right_roi_label = QLabel("NO SIGNAL")
        self.right_roi_label.setAlignment(Qt.AlignCenter)
        self.right_roi_label.setFixedSize(210, 80)
        self.right_roi_label.setStyleSheet(
            "background-color: #050B18; border-radius: 4px; color: #5E6E88; font-size: 9px;"
        )

        self.lbl_r_sig = QLabel("● GOOD SIGNAL")
        self.lbl_r_sig.setStyleSheet(
            "color: #39FF14; font-size: 8pt; font-weight: bold;"
        )
        self.lbl_r_sig.setAlignment(Qt.AlignCenter)

        right_eye_vbox.addWidget(lbl_right_title)
        right_eye_vbox.addWidget(self.right_roi_label, 0, Qt.AlignCenter)
        right_eye_vbox.addWidget(self.lbl_r_sig)
        roi_grid.addLayout(right_eye_vbox)

        roi_card_layout.addLayout(roi_grid)
        left_layout.addWidget(roi_card)

        body_layout.addLayout(left_layout, 5)

        # ------------------------------------------
        # RIGHT COLUMN: HEART RATE & STATS & GRAPHS
        # ------------------------------------------
        right_layout = QVBoxLayout()
        right_layout.setSpacing(12)

        # 1. HEART RATE CARD
        hr_card = QFrame()
        hr_card.setObjectName("card")
        hr_card.setMinimumHeight(
            135
        )  # Guarantee height to prevent vertical compression
        hr_card_layout = QVBoxLayout(hr_card)
        hr_card_layout.setContentsMargins(15, 6, 15, 6)
        hr_card_layout.setSpacing(4)

        # Header Row: Title & Heart icon
        hr_header = QHBoxLayout()
        lbl_hr_title = QLabel("HEART RATE")
        lbl_hr_title.setStyleSheet(
            "font-size: 8.5pt; color: #8F9CA3; font-weight: bold; letter-spacing: 0.5px;"
        )
        lbl_heart_icon = QLabel("❤️")
        lbl_heart_icon.setStyleSheet("font-size: 13pt;")
        hr_header.addWidget(lbl_hr_title)
        hr_header.addStretch()
        hr_header.addWidget(lbl_heart_icon)
        hr_card_layout.addLayout(hr_header)

        # Body Row: Big numbers & Stats
        hr_body = QHBoxLayout()

        # Left side: Big numbers
        bpm_vbox = QVBoxLayout()
        bpm_vbox.setSpacing(2)
        bpm_hbox = QHBoxLayout()
        bpm_hbox.setSpacing(8)

        self.bpm_label = QLabel("---")
        self.bpm_label.setFont(QFont("Consolas", 32, QFont.Bold))
        self.bpm_label.setStyleSheet("color: #39FF14;")
        self.bpm_label.setMinimumHeight(38)
        self.bpm_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        lbl_bpm_unit = QLabel("BPM")
        lbl_bpm_unit.setFont(QFont("Segoe UI", 11, QFont.Bold))
        lbl_bpm_unit.setStyleSheet("color: #39FF14; padding-bottom: 4px;")

        bpm_hbox.addWidget(self.bpm_label, 0, Qt.AlignLeft | Qt.AlignVCenter)
        bpm_hbox.addWidget(lbl_bpm_unit, 0, Qt.AlignLeft | Qt.AlignBottom)
        bpm_vbox.addLayout(bpm_hbox)

        self.lbl_heart_status = QLabel("STATUS: CALIBRATING...")
        self.lbl_heart_status.setStyleSheet(
            "font-size: 8.5pt; color: #FF9F0A; font-weight: bold;"
        )
        bpm_vbox.addWidget(self.lbl_heart_status, 0, Qt.AlignLeft)
        hr_body.addLayout(bpm_vbox)

        hr_body.addStretch()

        # Right side: Stats Grid
        hr_stats_grid = QGridLayout()
        hr_stats_grid.setSpacing(10)

        lbl_conf_title = QLabel("CONFIDENCE")
        lbl_conf_title.setStyleSheet(
            "font-size: 7.5pt; color: #8F9CA3; font-weight: bold;"
        )
        self.confidence_label = QLabel("-- %")
        self.confidence_label.setStyleSheet(
            "font-size: 10pt; color: #00FFFF; font-weight: bold; font-family: 'Consolas';"
        )

        lbl_quality_title = QLabel("SIGNAL QUALITY")
        lbl_quality_title.setStyleSheet(
            "font-size: 7.5pt; color: #8F9CA3; font-weight: bold;"
        )
        self.signal_quality_label = QLabel("GOOD")
        self.signal_quality_label.setStyleSheet(
            "font-size: 10pt; color: #39FF14; font-weight: bold; font-family: 'Consolas';"
        )

        hr_stats_grid.addWidget(lbl_conf_title, 0, 0)
        hr_stats_grid.addWidget(self.confidence_label, 0, 1)
        hr_stats_grid.addWidget(lbl_quality_title, 1, 0)
        hr_stats_grid.addWidget(self.signal_quality_label, 1, 1)

        hr_body.addLayout(hr_stats_grid)
        hr_card_layout.addLayout(hr_body)

        right_layout.addWidget(hr_card)

        # 2. PULSE WAVEFORM CARD
        pulse_card = QFrame()
        pulse_card.setObjectName("card")
        pulse_card_layout = QVBoxLayout(pulse_card)
        pulse_card_layout.setContentsMargins(12, 10, 12, 10)
        pulse_card_layout.setSpacing(6)

        lbl_pulse_header = QLabel("PULSE WAVEFORM")
        lbl_pulse_header.setStyleSheet(
            "font-size: 8.5pt; color: #FFFFFF; font-weight: bold; letter-spacing: 0.5px;"
        )
        pulse_card_layout.addWidget(lbl_pulse_header)

        self.pulse_graph = PulseGraph()
        self.pulse_graph.setMinimumHeight(100)
        pulse_card_layout.addWidget(self.pulse_graph)

        right_layout.addWidget(pulse_card)

        # 3. FREQUENCY SPECTRUM CARD
        spectrum_card = QFrame()
        spectrum_card.setObjectName("card")
        spectrum_card_layout = QVBoxLayout(spectrum_card)
        spectrum_card_layout.setContentsMargins(12, 10, 12, 10)
        spectrum_card_layout.setSpacing(6)

        lbl_spectrum_header = QLabel("FREQUENCY SPECTRUM")
        lbl_spectrum_header.setStyleSheet(
            "font-size: 8.5pt; color: #FFFFFF; font-weight: bold; letter-spacing: 0.5px;"
        )
        spectrum_card_layout.addWidget(lbl_spectrum_header)

        spectrum_body = QHBoxLayout()
        spectrum_body.setSpacing(10)

        self.spectrum_graph = SpectrumGraph()
        self.spectrum_graph.setMinimumHeight(100)
        spectrum_body.addWidget(self.spectrum_graph, 4)

        # Right stats sidebar inside spectrum card
        spectrum_stats = QVBoxLayout()
        spectrum_stats.setSpacing(10)
        spectrum_stats.setAlignment(Qt.AlignVCenter)

        lbl_dom_title = QLabel("DOMINANT FREQ.")
        lbl_dom_title.setStyleSheet(
            "font-size: 7.5pt; color: #8F9CA3; font-weight: bold;"
        )
        self.hz_label = QLabel("0.00 Hz")
        self.hz_label.setStyleSheet(
            "font-size: 10pt; color: #00FFFF; font-weight: bold; font-family: 'Consolas';"
        )

        lbl_pred_title = QLabel("PREDICTED BPM")
        lbl_pred_title.setStyleSheet(
            "font-size: 7.5pt; color: #8F9CA3; font-weight: bold;"
        )
        self.predicted_bpm_val = QLabel("--- BPM")
        self.predicted_bpm_val.setStyleSheet(
            "font-size: 10pt; color: #39FF14; font-weight: bold; font-family: 'Consolas';"
        )

        spectrum_stats.addWidget(lbl_dom_title)
        spectrum_stats.addWidget(self.hz_label)
        spectrum_stats.addWidget(lbl_pred_title)
        spectrum_stats.addWidget(self.predicted_bpm_val)

        spectrum_body.addLayout(spectrum_stats, 1)
        spectrum_card_layout.addLayout(spectrum_body)

        right_layout.addWidget(spectrum_card)

        # 4. SESSION STATISTICS CARD
        stats_card = QFrame()
        stats_card.setObjectName("card")
        stats_card_layout = QVBoxLayout(stats_card)
        stats_card_layout.setContentsMargins(15, 12, 15, 15)
        stats_card_layout.setSpacing(8)

        lbl_stats_header = QLabel("SESSION STATISTICS")
        lbl_stats_header.setStyleSheet(
            "font-size: 8.5pt; color: #FFFFFF; font-weight: bold; letter-spacing: 0.5px;"
        )
        stats_card_layout.addWidget(lbl_stats_header)

        stats_hbox = QHBoxLayout()
        stats_hbox.setSpacing(5)

        # Helper function to create individual metric layout
        def add_metric_col(title_text, val_widget):
            vbox = QVBoxLayout()
            vbox.setSpacing(4)
            vbox.setAlignment(Qt.AlignCenter)
            lbl = QLabel(title_text)
            lbl.setStyleSheet("font-size: 7pt; color: #8F9CA3; font-weight: bold;")
            lbl.setAlignment(Qt.AlignCenter)
            val_widget.setAlignment(Qt.AlignCenter)
            vbox.addWidget(lbl)
            vbox.addWidget(val_widget)
            return vbox

        self.lbl_duration = QLabel("00:00")
        self.lbl_duration.setFont(QFont("Consolas", 10, QFont.Bold))
        self.lbl_duration.setStyleSheet("color: #00FFFF;")

        self.lbl_avg_bpm = QLabel("--")
        self.lbl_avg_bpm.setFont(QFont("Consolas", 10, QFont.Bold))
        self.lbl_avg_bpm.setStyleSheet("color: #00FFFF;")

        self.lbl_min_bpm = QLabel("--")
        self.lbl_min_bpm.setFont(QFont("Consolas", 10, QFont.Bold))
        self.lbl_min_bpm.setStyleSheet("color: #00FFFF;")

        self.lbl_max_bpm = QLabel("--")
        self.lbl_max_bpm.setFont(QFont("Consolas", 10, QFont.Bold))
        self.lbl_max_bpm.setStyleSheet("color: #00FFFF;")

        self.lbl_stability = QLabel("GOOD")
        self.lbl_stability.setFont(QFont("Consolas", 10, QFont.Bold))
        self.lbl_stability.setStyleSheet("color: #39FF14;")

        self.lbl_tracking = QLabel("ACTIVE")
        self.lbl_tracking.setFont(QFont("Consolas", 10, QFont.Bold))
        self.lbl_tracking.setStyleSheet("color: #39FF14;")

        stats_hbox.addLayout(add_metric_col("DURATION", self.lbl_duration))
        stats_hbox.addLayout(add_metric_col("AVG BPM", self.lbl_avg_bpm))
        stats_hbox.addLayout(add_metric_col("MIN BPM", self.lbl_min_bpm))
        stats_hbox.addLayout(add_metric_col("MAX BPM", self.lbl_max_bpm))
        stats_hbox.addLayout(add_metric_col("STABILITY", self.lbl_stability))
        stats_hbox.addLayout(add_metric_col("TRACKING", self.lbl_tracking))

        stats_card_layout.addLayout(stats_hbox)
        right_layout.addWidget(stats_card)

        body_layout.addLayout(right_layout, 6)
        main_layout.addLayout(body_layout)

    def update_bpm_style(self, alert_level):
        """
        Updates the circular dial or borders if needed.
        In this card-based flat design, we don't need circular borders, but we can set
        accent colors for status text if needed.
        """
        pass

    def update_camera_frame(self, frame):
        """
        Displays BGR video frame onto camera label widget.
        """
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.camera_feed_label.setPixmap(
            QPixmap.fromImage(q_img).scaled(
                self.camera_feed_label.width(),
                self.camera_feed_label.height(),
                Qt.KeepAspectRatio,
            )
        )

    def update_eye_thumbnails(self, eye_rois):
        """
        Displays cropped eye thumbnails onto previews.
        """
        left_roi = eye_rois.get("left_roi")
        right_roi = eye_rois.get("right_roi")

        # Update Left Eye Crop
        if left_roi is not None and left_roi.size > 0:
            rgb_l = cv2.cvtColor(left_roi, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_l.shape
            q_img_l = QImage(rgb_l.data, w, h, ch * w, QImage.Format_RGB888)
            self.left_roi_label.setPixmap(
                QPixmap.fromImage(q_img_l).scaled(
                    self.left_roi_label.width() - 4,
                    self.left_roi_label.height() - 4,
                    Qt.KeepAspectRatio,
                )
            )
            self.lbl_l_sig.setText("● GOOD SIGNAL")
            self.lbl_l_sig.setStyleSheet(
                "color: #39FF14; font-size: 8pt; font-weight: bold;"
            )
        else:
            self.left_roi_label.setText("NO SIGNAL")
            self.lbl_l_sig.setText("● NO SIGNAL")
            self.lbl_l_sig.setStyleSheet(
                "color: #FF3B30; font-size: 8pt; font-weight: bold;"
            )

        # Update Right Eye Crop
        if right_roi is not None and right_roi.size > 0:
            rgb_r = cv2.cvtColor(right_roi, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_r.shape
            q_img_r = QImage(rgb_r.data, w, h, ch * w, QImage.Format_RGB888)
            self.right_roi_label.setPixmap(
                QPixmap.fromImage(q_img_r).scaled(
                    self.right_roi_label.width() - 4,
                    self.right_roi_label.height() - 4,
                    Qt.KeepAspectRatio,
                )
            )
            self.lbl_r_sig.setText("● GOOD SIGNAL")
            self.lbl_r_sig.setStyleSheet(
                "color: #39FF14; font-size: 8pt; font-weight: bold;"
            )
        else:
            self.right_roi_label.setText("NO SIGNAL")
            self.lbl_r_sig.setText("● NO SIGNAL")
            self.lbl_r_sig.setStyleSheet(
                "color: #FF3B30; font-size: 8pt; font-weight: bold;"
            )
