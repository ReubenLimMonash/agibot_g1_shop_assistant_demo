#!/usr/bin/env python3
"""
Flask webapp: displays three ROS2 camera streams in a responsive layout.
- /camera/head_color: main center display (large)
- /camera/hand_left_color: smaller panel (bottom left)
- /camera/hand_right_color: smaller panel (bottom right)

Usage:
  python3 flask_camera_display.py

Dependencies:
  pip install flask opencv-python numpy
  and a ROS2-capable Python environment with `rclpy` and `cv_bridge` (optional)
"""

from flask import Flask, Response
import threading
import time
import traceback
import signal
import sys

import numpy as np
import cv2

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
except Exception:
    rclpy = None

try:
    from cv_bridge import CvBridge
    _HAS_CV_BRIDGE = True
except Exception:
    CvBridge = None
    _HAS_CV_BRIDGE = False

# Shared state for latest frames and thread control
latest_frames = {
    'head_color': None,
    'hand_left_color': None,
    'hand_right_color': None,
}
frame_locks = {
    'head_color': threading.Lock(),
    'hand_left_color': threading.Lock(),
    'hand_right_color': threading.Lock(),
}
ros_thread = None

class ImageSubscriber(Node):
    def __init__(self, image_topics: dict):
        """
        Args:
            image_topics: dict mapping stream_name -> ROS2 topic path
        """
        super().__init__('flask_camera_display_sub')
        self.bridge = CvBridge() if _HAS_CV_BRIDGE else None
        self.stream_names = {}
        
        for stream_name, topic in image_topics.items():
            self.create_subscription(Image, topic, self.make_callback(stream_name), 10)
            self.stream_names[topic] = stream_name

    def make_callback(self, stream_name: str):
        """Create a callback for a specific stream."""
        def cb(msg: Image):
            try:
                if self.bridge is not None:
                    cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                else:
                    arr = np.frombuffer(msg.data, dtype=np.uint8)
                    if msg.encoding in ('rgb8', 'bgr8'):
                        arr = arr.reshape((msg.height, msg.width, 3))
                        if msg.encoding == 'rgb8':
                            cv_img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                        else:
                            cv_img = arr
                    elif msg.encoding == 'mono8':
                        cv_img = arr.reshape((msg.height, msg.width))
                    else:
                        # best effort
                        try:
                            arr = arr.reshape((msg.height, msg.width, 3))
                            cv_img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                        except Exception:
                            return

                with frame_locks[stream_name]:
                    latest_frames[stream_name] = cv_img.copy()
            except Exception:
                traceback.print_exc()
        
        return cb


def ros_thread_main(image_topics: dict):
    """
    Args:
        image_topics: dict mapping stream_name -> ROS2 topic path
    """
    if rclpy is None:
        print('rclpy not available in this Python. ROS subscriber disabled.')
        return
    try:
        rclpy.init()
        node = ImageSubscriber(image_topics=image_topics)
        print('ROS subscriber node started for topics:', list(image_topics.values()))
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()
    except Exception:
        traceback.print_exc()


def get_jpeg_frame(stream_name: str):
    """Return latest frame for a stream encoded as JPEG bytes, or None if no frame yet."""
    with frame_locks[stream_name]:
        frame = None if latest_frames[stream_name] is None else latest_frames[stream_name].copy()
    if frame is None:
        return None
    # Encode as JPEG
    ok, buf = cv2.imencode('.jpg', frame)
    if not ok:
        return None
    return buf.tobytes()


app = Flask(__name__)

# Configuration: map stream names to ROS2 topics
CAMERA_TOPICS = {
    'head_color': '/camera/head_color',
    'hand_left_color': '/camera/hand_left_color',
    'hand_right_color': '/camera/hand_right_color',
}


@app.route('/')
def index():
    """Serve the main page with three video streams."""
    html = '''
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Camera Display</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
          body {
            background-color: #f8f9fa;
          }
          .camera-container {
            margin: 20px 0;
            text-align: center;
          }
          .main-camera {
            max-width: 100%;
            border: 2px solid #007bff;
            border-radius: 8px;
            margin-bottom: 20px;
          }
          .sub-camera {
            max-width: 100%;
            border: 2px solid #6c757d;
            border-radius: 8px;
          }
          .camera-label {
            font-weight: bold;
            margin-top: 10px;
            color: #333;
          }
          .sub-cameras-row {
            display: flex;
            gap: 20px;
            justify-content: center;
            flex-wrap: wrap;
          }
          .sub-camera-panel {
            flex: 1;
            min-width: 300px;
            max-width: 450px;
          }
        </style>
      </head>
      <body>
        <div class="container-fluid">
          <h1 class="mt-4 mb-4">Camera Streams</h1>
          
          <!-- Main camera (head) -->
          <div class="camera-container">
            <div class="camera-label">Head Camera</div>
            <img id="head-camera" src="/video_feed/head_color" class="main-camera" alt="Head Camera" />
          </div>
          
          <!-- Sub cameras (left and right hands) -->
          <div class="sub-cameras-row">
            <div class="sub-camera-panel">
              <div class="camera-label">Left Hand Camera</div>
              <img id="left-camera" src="/video_feed/hand_left_color" class="sub-camera" alt="Left Hand Camera" />
            </div>
            <div class="sub-camera-panel">
              <div class="camera-label">Right Hand Camera</div>
              <img id="right-camera" src="/video_feed/hand_right_color" class="sub-camera" alt="Right Hand Camera" />
            </div>
          </div>
        </div>
      </body>
    </html>
    '''
    return html


@app.route('/video_feed/<stream_name>')
def video_feed(stream_name):
    """Stream video frames for the specified camera."""
    if stream_name not in CAMERA_TOPICS:
        return "Invalid stream", 404
    
    def generate():
        while True:
            frame = get_jpeg_frame(stream_name)
            if frame is None:
                # send a tiny placeholder image every 0.2s to keep connection alive
                blank = np.zeros((240, 320, 3), dtype=np.uint8)
                ok, buf = cv2.imencode('.jpg', blank)
                if ok:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
                time.sleep(0.2)
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


def start_ros_thread():
    """Start the ROS subscriber thread."""
    global ros_thread
    if ros_thread is None or not ros_thread.is_alive():
        ros_thread = threading.Thread(
            target=ros_thread_main, 
            args=(CAMERA_TOPICS,), 
            daemon=True
        )
        ros_thread.start()


def stop_ros_thread():
    """Stop the ROS subscriber thread."""
    global ros_thread
    if ros_thread is not None:
        ros_thread.join(timeout=1.0)


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print('\nShutting down gracefully...')
    stop_ros_thread()
    sys.exit(0)


if __name__ == '__main__':
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # start ROS subscriber thread
        start_ros_thread()
        # start flask app
        app.run(host='0.0.0.0', port=8500, threaded=True)
    except KeyboardInterrupt:
        print('\nShutting down gracefully...')
        stop_ros_thread()
    except Exception as e:
        print(f'Error: {e}')
        traceback.print_exc()
        stop_ros_thread()
    finally:
        print('Shutdown complete.')
        sys.exit(0)
