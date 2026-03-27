from flask import Flask, render_template, Response, jsonify, request
import cv2
from utils.rppg_processing import RPPGMonitor

app = Flask(__name__)

# Live webcam
camera = cv2.VideoCapture(0)

# Dataset sample video
sample_video_path = "dataset/VideoCutter_20260327191553567.mp4""
sample_video = cv2.VideoCapture(sample_video_path)

monitor = RPPGMonitor()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    def generate_frames():
        global camera, sample_video, monitor

        while True:
            # Demo mode -> dataset sample video
            if monitor.demo_mode:
                success, frame = sample_video.read()

                # If video ends, restart it
                if not success:
                    sample_video.release()
                    sample_video = cv2.VideoCapture(sample_video_path)
                    continue

                processed_frame = monitor.process_frame(frame, source="video")

                ret, buffer = cv2.imencode(".jpg", processed_frame)
                frame_bytes = buffer.tobytes()

                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
                continue

            # Live mode -> webcam
            success, frame = camera.read()
            if not success:
                break

            processed_frame = monitor.process_frame(frame, source="live")

            ret, buffer = cv2.imencode(".jpg", processed_frame)
            frame_bytes = buffer.tobytes()

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stats")
def stats():
    return jsonify(monitor.get_stats())

@app.route("/analysis_data")
def analysis_data():
    return jsonify(monitor.get_analysis_data())

@app.route("/reset", methods=["POST"])
def reset():
    monitor.reset()
    return jsonify({"message": "Session reset done"})

@app.route("/demo_mode", methods=["POST"])
def demo_mode():
    data = request.get_json()
    value = data.get("enabled", False)
    monitor.set_demo_mode(value)
    return jsonify({"demo_mode": monitor.demo_mode})

if __name__ == "__main__":
    app.run(debug=True)
