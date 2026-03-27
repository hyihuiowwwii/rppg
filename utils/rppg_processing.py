import cv2
import numpy as np
import time
from collections import deque

class RPPGMonitor:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.signal_values = deque(maxlen=300)
        self.signal_times = deque(maxlen=300)
        self.bpm_values = deque(maxlen=100)
        self.bpm_times = deque(maxlen=100)

        self.current_bpm = 0
        self.demo_mode = False
        self.start_time = time.time()

    def reset(self):
        self.signal_values.clear()
        self.signal_times.clear()
        self.bpm_values.clear()
        self.bpm_times.clear()
        self.current_bpm = 0
        self.start_time = time.time()

    def set_demo_mode(self, value):
        self.demo_mode = value
        self.reset()

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        status_text = "Face not found"

        for (x, y, w, h) in faces:
            # Main face box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 220, 80), 2)

            # Forehead ROI
            fx1 = x + int(w * 0.30)
            fy1 = y + int(h * 0.12)
            fx2 = x + int(w * 0.70)
            fy2 = y + int(h * 0.28)

            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 255), 2)

            roi = frame[fy1:fy2, fx1:fx2]

            if roi.size != 0:
                green_mean = float(np.mean(roi[:, :, 1]))
                current_time = time.time() - self.start_time

                self.signal_values.append(green_mean)
                self.signal_times.append(current_time)

                if len(self.signal_values) >= 120:
                    bpm = self.estimate_bpm()
                    if bpm > 0:
                        self.current_bpm = bpm
                        self.bpm_values.append(bpm)
                        self.bpm_times.append(current_time)

            status_text = "Face detected"
            break

        cv2.putText(frame, f"BPM: {self.current_bpm}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, status_text, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame

    def estimate_bpm(self):
        signal = np.array(self.signal_values, dtype=np.float32)
        times = np.array(self.signal_times, dtype=np.float32)

        if len(signal) < 120:
            return 0

        # Remove trend
        signal = signal - np.mean(signal)

        duration = times[-1] - times[0]
        if duration <= 0:
            return 0

        fs = len(signal) / duration
        if fs < 5:
            return 0

        freqs = np.fft.rfftfreq(len(signal), d=1/fs)
        fft_signal = np.abs(np.fft.rfft(signal))

        # Human heart rate range: 48 BPM to 180 BPM
        valid = (freqs >= 0.8) & (freqs <= 3.0)

        if np.sum(valid) == 0:
            return 0

        peak_freq = freqs[valid][np.argmax(fft_signal[valid])]
        bpm = int(peak_freq * 60)

        if 48 <= bpm <= 180:
            return bpm

        return 0

    def get_stats(self):
        bpm_list = list(self.bpm_values)

        if len(bpm_list) == 0:
            return {
                "current_bpm": self.current_bpm,
                "avg_bpm": 0,
                "min_bpm": 0,
                "max_bpm": 0,
                "samples": len(self.signal_values),
                "mode": "Demo" if self.demo_mode else "Live"
            }

        return {
            "current_bpm": self.current_bpm,
            "avg_bpm": round(sum(bpm_list) / len(bpm_list), 2),
            "min_bpm": min(bpm_list),
            "max_bpm": max(bpm_list),
            "samples": len(self.signal_values),
            "mode": "Demo" if self.demo_mode else "Live"
        }

    def get_analysis_data(self):
        if self.demo_mode:
            return self.get_demo_analysis_data()

        return {
            "signal_times": list(self.signal_times),
            "signal_values": list(self.signal_values),
            "bpm_times": list(self.bpm_times),
            "bpm_values": list(self.bpm_values)
        }

    def get_demo_analysis_data(self):
        t = np.linspace(0, 20, 200)
        bpm_center = 78
        freq = bpm_center / 60.0

        raw_signal = 20 * np.sin(2 * np.pi * freq * t) + 100 + np.random.normal(0, 1.5, len(t))
        bpm_curve = 78 + 2 * np.sin(0.5 * t)

        return {
            "signal_times": t.tolist(),
            "signal_values": raw_signal.tolist(),
            "bpm_times": t.tolist(),
            "bpm_values": bpm_curve.tolist()
        }

    def get_demo_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (30, 35, 45)

        cv2.putText(frame, "Demo Mode Active", (170, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.rectangle(frame, (220, 120), (420, 340), (50, 220, 80), 2)
        cv2.rectangle(frame, (280, 150), (360, 190), (0, 255, 255), 2)

        fake_bpm = 78
        self.current_bpm = fake_bpm
        current_time = time.time() - self.start_time

        self.bpm_values.append(fake_bpm)
        self.bpm_times.append(current_time)

        cv2.putText(frame, f"BPM: {fake_bpm}", (220, 390),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, "Using sample signal for presentation", (130, 430),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame
