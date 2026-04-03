"""
modules/video_stream.py
------------------------
Thread-safe video capture handler for webcam/CCTV/file sources.
Produces MJPEG frames for Flask streaming and provides frame access
for the processing pipeline.
"""

from __future__ import annotations

import threading
import time
import cv2
import numpy as np
from typing import Optional

from utils.logger import get_logger

logger = get_logger("VideoStream")


class VideoStream:
    """Thread-safe video stream handler.

    Captures frames from a webcam, CCTV camera, or video file in a
    background thread. Provides the latest frame on demand.

    Args:
        source: Video source (0 for webcam, path for file, URL for CCTV).
        width:  Desired frame width.
        height: Desired frame height.
    """

    def __init__(
        self,
        source=0,
        width: int = 640,
        height: int = 480,
    ):
        self.source = source
        self.width  = width
        self.height = height

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_count = 0
        self._fps = 0.0
        self._fps_timer = time.perf_counter()
        self._fps_count = 0

    def start(self) -> "VideoStream":
        """Open the video source and start the capture thread."""
        logger.info("Opening video source: %s", self.source)
        self._cap = cv2.VideoCapture(self.source)

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")

        # Try to set resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps  = self._cap.get(cv2.CAP_PROP_FPS) or 30.0

        logger.info("Source opened: %dx%d @ %.1f FPS", actual_w, actual_h, src_fps)

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        return self

    def _capture_loop(self):
        """Background thread that continuously reads frames."""
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                break

            ret, frame = self._cap.read()
            if not ret:
                # For video files, loop back to start
                if isinstance(self.source, str):
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            # Resize if needed
            h, w = frame.shape[:2]
            if w != self.width or h != self.height:
                frame = cv2.resize(frame, (self.width, self.height))

            with self._lock:
                self._frame = frame
                self._frame_count += 1

            # FPS calculation
            self._fps_count += 1
            elapsed = time.perf_counter() - self._fps_timer
            if elapsed >= 1.0:
                self._fps = self._fps_count / elapsed
                self._fps_count = 0
                self._fps_timer = time.perf_counter()

            time.sleep(0.01)  # ~100 FPS max capture rate

    def read(self):
        """Get the latest frame.

        Returns:
            (success: bool, frame: np.ndarray or None)
        """
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame (convenience method)."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    @property
    def is_open(self) -> bool:
        return self._running and self._cap is not None and self._cap.isOpened()

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def stop(self):
        """Stop the capture thread and release the video source."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("Video stream stopped.")

    def release(self):
        """Alias for stop()."""
        self.stop()

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()


class FileVideoProcessor:
    """Process a video file frame-by-frame (non-threaded).

    Used for uploaded video analysis where we process sequentially.

    Args:
        filepath: Path to the video file.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._cap = cv2.VideoCapture(filepath)

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {filepath}")

        self.width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps    = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info("Opened video: %s (%dx%d, %d frames, %.1f FPS)",
                    filepath, self.width, self.height, self.total_frames, self.fps)

    def read(self):
        """Read the next frame."""
        ret, frame = self._cap.read()
        return ret, frame

    def release(self):
        if self._cap is not None:
            self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()
