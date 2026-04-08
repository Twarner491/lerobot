# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file contains utilities for recording frames from cameras. For more info look at `OpenCVCamera` docstring.
"""

import argparse
import concurrent.futures
import math
import platform
import shutil
import threading
import time
from pathlib import Path
from threading import Thread

import numpy as np
from PIL import Image

from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
from lerobot.common.robot_devices.utils import (
    RobotDeviceAlreadyConnectedError,
    RobotDeviceNotConnectedError,
    busy_wait,
)
from lerobot.common.utils.utils import capture_timestamp_utc

# The maximum opencv device index depends on your operating system. For instance,
# if you have 3 cameras, they should be associated to index 0, 1, and 2. This is the case
# on MacOS. However, on Ubuntu, the indices are different like 6, 16, 23.
# When you change the USB port or reboot the computer, the operating system might
# treat the same cameras as new devices. Thus we select a higher bound to search indices.
MAX_OPENCV_INDEX = 60


def find_cameras(raise_when_empty=False, max_index_search_range=MAX_OPENCV_INDEX, mock=False) -> list[dict]:
    cameras = []
    if platform.system() == "Linux":
        print("Linux detected. Finding available camera indices through scanning '/dev/video*' ports")
        possible_ports = [str(port) for port in Path("/dev").glob("video*")]
        ports = _find_cameras(possible_ports, mock=mock)
        for port in ports:
            cameras.append(
                {
                    "port": port,
                    "index": int(port.removeprefix("/dev/video")),
                }
            )
    else:
        print(
            "Mac or Windows detected. Finding available camera indices through "
            f"scanning all indices from 0 to {MAX_OPENCV_INDEX}"
        )
        possible_indices = range(max_index_search_range)
        indices = _find_cameras(possible_indices, mock=mock)
        for index in indices:
            cameras.append(
                {
                    "port": None,
                    "index": index,
                }
            )

    return cameras


def _find_cameras(
    possible_camera_ids: list[int | str], raise_when_empty=False, mock=False
) -> list[int | str]:
    if mock:
        import tests.cameras.mock_cv2 as cv2
    else:
        import cv2

    camera_ids = []
    for camera_idx in possible_camera_ids:
        camera = cv2.VideoCapture(camera_idx)
        is_open = camera.isOpened()
        camera.release()

        if is_open:
            print(f"Camera found at index {camera_idx}")
            camera_ids.append(camera_idx)

    if raise_when_empty and len(camera_ids) == 0:
        raise OSError(
            "Not a single camera was detected. Try re-plugging, or re-installing `opencv2`, "
            "or your camera driver, or make sure your camera is compatible with opencv2."
        )

    return camera_ids


def is_valid_unix_path(path: str) -> bool:
    """Note: if 'path' points to a symlink, this will return True only if the target exists"""
    p = Path(path)
    return p.is_absolute() and p.exists()


def get_camera_index_from_unix_port(port: Path) -> int:
    return int(str(port.resolve()).removeprefix("/dev/video"))


def save_image(img_array, camera_index, frame_index, images_dir):
    img = Image.fromarray(img_array)
    path = images_dir / f"camera_{camera_index:02d}_frame_{frame_index:06d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), quality=100)


def save_images_from_cameras(
    images_dir: Path,
    camera_ids: list | None = None,
    fps=None,
    width=None,
    height=None,
    record_time_s=2,
    mock=False,
):
    """
    Initializes all the cameras and saves images to the directory. Useful to visually identify the camera
    associated to a given camera index.
    """
    if camera_ids is None or len(camera_ids) == 0:
        camera_infos = find_cameras(mock=mock)
        camera_ids = [cam["index"] for cam in camera_infos]

    print("Connecting cameras")
    cameras = []
    for cam_idx in camera_ids:
        config = OpenCVCameraConfig(camera_index=cam_idx, fps=fps, width=width, height=height, mock=mock)
        camera = OpenCVCamera(config)
        camera.connect()
        print(
            f"OpenCVCamera({camera.camera_index}, fps={camera.fps}, width={camera.capture_width}, "
            f"height={camera.capture_height}, color_mode={camera.color_mode})"
        )
        cameras.append(camera)

    images_dir = Path(images_dir)
    if images_dir.exists():
        shutil.rmtree(
            images_dir,
        )
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving images to {images_dir}")
    frame_index = 0
    start_time = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        while True:
            now = time.perf_counter()

            for camera in cameras:
                # If we use async_read when fps is None, the loop will go full speed, and we will endup
                # saving the same images from the cameras multiple times until the RAM/disk is full.
                image = camera.read() if fps is None else camera.async_read()

                executor.submit(
                    save_image,
                    image,
                    camera.camera_index,
                    frame_index,
                    images_dir,
                )

            if fps is not None:
                dt_s = time.perf_counter() - now
                busy_wait(1 / fps - dt_s)

            print(f"Frame: {frame_index:04d}\tLatency (ms): {(time.perf_counter() - now) * 1000:.2f}")

            if time.perf_counter() - start_time > record_time_s:
                break

            frame_index += 1

    print(f"Images have been saved to {images_dir}")


class OpenCVCamera:
    """
    The OpenCVCamera class allows to efficiently record images from cameras. It relies on opencv2 to communicate
    with the cameras. Most cameras are compatible. For more info, see the [Video I/O with OpenCV Overview](https://docs.opencv.org/4.x/d0/da7/videoio_overview.html).

    An OpenCVCamera instance requires a camera index (e.g. `OpenCVCamera(camera_index=0)`). When you only have one camera
    like a webcam of a laptop, the camera index is expected to be 0, but it might also be very different, and the camera index
    might change if you reboot your computer or re-plug your camera. This behavior depends on your operation system.

    To find the camera indices of your cameras, you can run our utility script that will be save a few frames for each camera:
    ```bash
    python lerobot/common/robot_devices/cameras/opencv.py --images-dir outputs/images_from_opencv_cameras
    ```

    When an OpenCVCamera is instantiated, if no specific config is provided, the default fps, width, height and color_mode
    of the given camera will be used.

    Example of usage:
    ```python
    from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig

    config = OpenCVCameraConfig(camera_index=0)
    camera = OpenCVCamera(config)
    camera.connect()
    color_image = camera.read()
    # when done using the camera, consider disconnecting
    camera.disconnect()
    ```

    Example of changing default fps, width, height and color_mode:
    ```python
    config = OpenCVCameraConfig(camera_index=0, fps=30, width=1280, height=720)
    config = OpenCVCameraConfig(camera_index=0, fps=90, width=640, height=480)
    config = OpenCVCameraConfig(camera_index=0, fps=90, width=640, height=480, color_mode="bgr")
    # Note: might error out open `camera.connect()` if these settings are not compatible with the camera
    ```
    """

    def __init__(self, config: OpenCVCameraConfig):
        self.config = config
        self.camera_index = config.camera_index
        self.port = None

        # Linux uses ports for connecting to cameras
        if platform.system() == "Linux":
            if isinstance(self.camera_index, int):
                self.port = Path(f"/dev/video{self.camera_index}")
            elif isinstance(self.camera_index, str) and is_valid_unix_path(self.camera_index):
                self.port = Path(self.camera_index)
                # Retrieve the camera index from a potentially symlinked path
                self.camera_index = get_camera_index_from_unix_port(self.port)
            else:
                raise ValueError(f"Please check the provided camera_index: {self.camera_index}")

        # Store the raw (capture) resolution from the config.
        self.capture_width = config.width
        self.capture_height = config.height

        # If rotated by ±90, swap width and height.
        if config.rotation in [-90, 90]:
            self.width = config.height
            self.height = config.width
        else:
            self.width = config.width
            self.height = config.height

        self.fps = config.fps
        self.channels = config.channels
        self.color_mode = config.color_mode
        self.mock = config.mock

        self.camera = None
        self.is_connected = False
        self.thread = None
        self.stop_event = None
        self.color_image = None
        self.logs = {}

        if self.mock:
            import tests.cameras.mock_cv2 as cv2
        else:
            import cv2

        self.rotation = None
        if config.rotation == -90:
            self.rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
        elif config.rotation == 90:
            self.rotation = cv2.ROTATE_90_CLOCKWISE
        elif config.rotation == 180:
            self.rotation = cv2.ROTATE_180

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(f"OpenCVCamera({self.camera_index}) is already connected.")

        if self.mock:
            import tests.cameras.mock_cv2 as cv2
        else:
            import cv2

            # Use 1 thread to avoid blocking the main thread. Especially useful during data collection
            # when other threads are used to save the images.
            cv2.setNumThreads(1)

        backend = (
            cv2.CAP_V4L2
            if platform.system() == "Linux"
            else cv2.CAP_DSHOW
            if platform.system() == "Windows"
            else cv2.CAP_AVFOUNDATION
            if platform.system() == "Darwin"
            else cv2.CAP_ANY
        )

        camera_idx = f"/dev/video{self.camera_index}" if platform.system() == "Linux" and isinstance(self.camera_index, int) else self.camera_index
        
        # Critical: For ArduCam camera (video0), force MJPG format
        is_arducam = isinstance(self.camera_index, str) and "video0" in self.camera_index
        
        # For ArduCam, set format before opening
        if is_arducam and platform.system() == "Linux":
            try:
                # Set the camera format to MJPG using v4l2-ctl (more reliable than OpenCV properties)
                import subprocess
                subprocess.run(["v4l2-ctl", "-d", "/dev/video0", "--set-fmt-video=width=640,height=480,pixelformat=MJPG"], 
                           check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print("Set ArduCam format to MJPG using v4l2-ctl")
            except Exception as e:
                print(f"Warning: Could not set format with v4l2-ctl: {e}")

        # First create a temporary camera trying to access `camera_index`,
        # and verify it is a valid camera by calling `isOpened`.
        tmp_camera = cv2.VideoCapture(camera_idx, backend)
        is_camera_open = tmp_camera.isOpened()
        # Release camera to make it accessible
        tmp_camera.release()
        del tmp_camera

        # If the camera doesn't work, display the camera indices corresponding to
        # valid cameras.
        if not is_camera_open:
            # Verify that the provided `camera_index` is valid before printing the traceback
            cameras_info = find_cameras()
            available_cam_ids = [cam["index"] for cam in cameras_info]
            if self.camera_index not in available_cam_ids:
                raise ValueError(
                    f"`camera_index` is expected to be one of these available cameras {available_cam_ids}, but {self.camera_index} is provided instead. "
                    "To find the camera index you should use, run `python lerobot/common/robot_devices/cameras/opencv.py`."
                )

            raise OSError(f"Can't access OpenCVCamera({camera_idx}).")

        # Secondly, create the camera that will be used downstream.
        self.camera = cv2.VideoCapture(camera_idx, backend)
        
        # For ArduCam, must set MJPG format BEFORE setting resolution and fps
        if is_arducam:
            # Force MJPG format for ArduCam (essential for it to work properly)
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            self.camera.set(cv2.CAP_PROP_FOURCC, fourcc)
            print(f"Setting ArduCam camera format to MJPG")

        if self.fps is not None:
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)
        if self.capture_width is not None:
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
        if self.capture_height is not None:
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
            
        # Additional camera settings for Arducam USB camera
        if is_arducam:
            print(f"Applying ArduCam-specific settings for {self.camera_index}")
            try:
                # Buffer size must be minimal to prevent lag
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Set specific camera parameters from testMS.py
                self.camera.set(cv2.CAP_PROP_BRIGHTNESS, 0)     # Default 0, range -64 to 64
                self.camera.set(cv2.CAP_PROP_CONTRAST, 32)      # Default 32, range 0 to 64
                self.camera.set(cv2.CAP_PROP_SATURATION, 64)    # Default 64, range 0 to 128
                self.camera.set(cv2.CAP_PROP_GAIN, 0)           # Default 0, range 0 to 100
                
                # Test read with a short timeout
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("Camera read timed out")
                
                # Set timeout for 0.5 seconds
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(1)
                
                try:
                    ret, _ = self.camera.read()
                    signal.alarm(0)  # Cancel alarm
                    if not ret:
                        print(f"Warning: Initial frame read failed for {camera_idx}")
                except TimeoutError:
                    signal.alarm(0)  # Cancel alarm
                    print(f"Warning: Initial frame read timed out for {camera_idx}")
                    # Recreate the camera - sometimes this helps
                    self.camera.release()
                    self.camera = cv2.VideoCapture(camera_idx, backend)
                    self.camera.set(cv2.CAP_PROP_FOURCC, fourcc)
                    if self.fps is not None:
                        self.camera.set(cv2.CAP_PROP_FPS, self.fps)
                    if self.capture_width is not None:
                        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
                    if self.capture_height is not None:
                        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
            except Exception as e:
                print(f"Warning: Could not apply ArduCam settings: {e}")

        actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
        actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        # Get the actual format
        try:
            fourcc_int = int(self.camera.get(cv2.CAP_PROP_FOURCC))
            fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
            
            # If this is ArduCam and not using MJPG, try to force it again
            if is_arducam and fourcc_str != "MJPG":
                print(f"Warning: ArduCam not using MJPG format (got {fourcc_str}). Attempting to force MJPG...")
                self.camera.release()
                
                # Try with direct format setting again
                import subprocess
                subprocess.run(["v4l2-ctl", "-d", "/dev/video0", "--set-fmt-video=width=640,height=480,pixelformat=MJPG"], 
                           check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                self.camera = cv2.VideoCapture(camera_idx, backend)
                fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                self.camera.set(cv2.CAP_PROP_FOURCC, fourcc)
                
                # Set other properties again
                if self.fps is not None:
                    self.camera.set(cv2.CAP_PROP_FPS, self.fps)
                if self.capture_width is not None:
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
                if self.capture_height is not None:
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
                
                # Get the format again to check
                fourcc_int = int(self.camera.get(cv2.CAP_PROP_FOURCC))
                fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
        except Exception as e:
            print(f"Error getting camera format: {e}")
            fourcc_str = "Unknown"

        # Using `math.isclose` since actual fps can be a float (e.g. 29.9 instead of 30)
        if self.fps is not None and not math.isclose(self.fps, actual_fps, rel_tol=1e-3):
            # For ArduCam, we'll be more lenient about FPS requirements
            if not is_arducam:
                raise OSError(
                    f"Can't set {self.fps=} for OpenCVCamera({self.camera_index}). Actual value is {actual_fps}."
                )
            else:
                print(f"Warning: ArduCam FPS set to {self.fps} but actual is {actual_fps}. Continuing anyway.")
                
        if self.capture_width is not None and not math.isclose(
            self.capture_width, actual_width, rel_tol=1e-3
        ):
            if not is_arducam:
                raise OSError(
                    f"Can't set {self.capture_width=} for OpenCVCamera({self.camera_index}). Actual value is {actual_width}."
                )
            else:
                print(f"Warning: ArduCam width set to {self.capture_width} but actual is {actual_width}. Continuing anyway.")
                
        if self.capture_height is not None and not math.isclose(
            self.capture_height, actual_height, rel_tol=1e-3
        ):
            if not is_arducam:
                raise OSError(
                    f"Can't set {self.capture_height=} for OpenCVCamera({self.camera_index}). Actual value is {actual_height}."
                )
            else:
                print(f"Warning: ArduCam height set to {self.capture_height} but actual is {actual_height}. Continuing anyway.")

        self.fps = round(actual_fps) if actual_fps > 0 else 30
        self.capture_width = round(actual_width)
        self.capture_height = round(actual_height)
        self.is_connected = True
        
        # Print device info for debugging
        print(f"Camera {self.camera_index} ready:")
        print(f"  Resolution: {self.capture_width}x{self.capture_height}")
        print(f"  FPS: {self.fps}")
        print(f"  Format: {fourcc_str}")

        # For ArduCam, perform a test read to ensure it's working
        if is_arducam:
            # Read a few frames to "warm up" the camera
            for _ in range(3):
                ret, _ = self.camera.read()
                if ret:
                    print("ArduCam successfully delivered a test frame")
                    break

    def read(self, temporary_color_mode: str | None = None) -> np.ndarray:
        """Read a frame from the camera returned in the format (height, width, channels)
        (e.g. 480 x 640 x 3), contrarily to the pytorch format which is channel first.

        Note: Reading a frame is done every `camera.fps` times per second, and it is blocking.
        If you are reading data from other sensors, we advise to use `camera.async_read()` which is non blocking version of `camera.read()`.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"OpenCVCamera({self.camera_index}) is not connected. Try running `camera.connect()` first."
            )

        start_time = time.perf_counter()
        
        # Check if this is the ArduCam camera
        is_arducam = isinstance(self.camera_index, str) and "video0" in self.camera_index

        # Add retry logic for Arducam camera which may occasionally fail
        max_retries = 5 if is_arducam else 1
        retry_count = 0
        
        while retry_count < max_retries:
            ret, color_image = self.camera.read()
            
            if ret:
                break
                
            # If we got a bad frame, retry for Arducam camera
            if is_arducam:
                print(f"Retrying read for ArduCam camera ({retry_count+1}/{max_retries})")
                time.sleep(0.05)  # Short pause before retrying
                retry_count += 1
            else:
                # For non-Arducam cameras, don't retry
                break
        
        if not ret:
            # For the Arducam camera, return the last successful image if available
            # rather than crashing the program
            if is_arducam and self.color_image is not None:
                print(f"Warning: Using last good frame for ArduCam camera")
                return self.color_image
            
            raise OSError(f"Can't capture color image from camera {self.camera_index}.")

        # Import cv2 here to avoid circular import issues
        if self.mock:
            import tests.cameras.mock_cv2 as cv2
        else:
            import cv2
            
        # Enhance ArduCam images if they are too dark (like in testMS.py)
        if is_arducam:
            # Check brightness
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            
            # If image is too dark, enhance it
            if brightness < 70:
                # Increase brightness
                alpha = 1.75  # Contrast control
                beta = 40     # Brightness control
                color_image = cv2.convertScaleAbs(color_image, alpha=alpha, beta=beta)
                
                # Apply CLAHE for better contrast in dark areas
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                lab = cv2.cvtColor(color_image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l = clahe.apply(l)
                lab = cv2.merge((l, a, b))
                color_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        requested_color_mode = self.color_mode if temporary_color_mode is None else temporary_color_mode

        if requested_color_mode not in ["rgb", "bgr"]:
            raise ValueError(
                f"Expected color values are 'rgb' or 'bgr', but {requested_color_mode} is provided."
            )

        # OpenCV uses BGR format as default (blue, green, red) for all operations, including displaying images.
        # However, Deep Learning framework such as LeRobot uses RGB format as default to train neural networks,
        # so we convert the image color from BGR to RGB.
        if requested_color_mode == "rgb":
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        h, w, _ = color_image.shape
        if h != self.capture_height or w != self.capture_width:
            if is_arducam:
                # For ArduCam, resize the image rather than failing
                print(f"Warning: ArduCam image size {w}x{h} doesn't match expected {self.capture_width}x{self.capture_height}. Resizing...")
                color_image = cv2.resize(color_image, (self.capture_width, self.capture_height))
            else:
                raise OSError(
                    f"Can't capture color image with expected height and width ({self.capture_height} x {self.capture_width}). ({h} x {w}) returned instead."
                )

        if self.rotation is not None:
            color_image = cv2.rotate(color_image, self.rotation)

        # log the number of seconds it took to read the image
        self.logs["delta_timestamp_s"] = time.perf_counter() - start_time

        # log the utc time at which the image was received
        self.logs["timestamp_utc"] = capture_timestamp_utc()

        self.color_image = color_image

        return color_image

    def read_loop(self):
        """Background thread that continuously reads frames from the camera"""
        # Check if this is the ArduCam camera
        is_arducam = isinstance(self.camera_index, str) and "video0" in self.camera_index
        
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        while not self.stop_event.is_set():
            try:
                self.color_image = self.read()
                consecutive_errors = 0  # Reset error counter on success
                # Short sleep to prevent CPU hogging
                time.sleep(0.01)
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors > max_consecutive_errors and is_arducam:
                    # For ArduCam, if we get too many errors, create a fallback image
                    print(f"Too many consecutive errors with ArduCam camera: {e}")
                    # Import here to avoid circular imports
                    import cv2
                    placeholder = np.zeros((self.capture_height, self.capture_width, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "ArduCam Error", 
                               (self.capture_width//4, self.capture_height//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    if self.color_mode == "rgb":
                        placeholder = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
                    self.color_image = placeholder
                else:
                    print(f"Error reading in thread: {e}")
                # Longer sleep on error to avoid rapid error loops
                time.sleep(0.1)

    def async_read(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"OpenCVCamera({self.camera_index}) is not connected. Try running `camera.connect()` first."
            )

        # Check if this is the ArduCam camera
        is_arducam = isinstance(self.camera_index, str) and "video0" in self.camera_index
        
        # Start the camera read thread if not already running
        if self.thread is None:
            self.stop_event = threading.Event()
            self.thread = Thread(target=self.read_loop, args=())
            self.thread.daemon = True
            self.thread.start()

        # Set appropriate timeout parameters for different cameras
        if is_arducam:
            # ArduCam needs more time and more leniency
            max_tries = self.fps * 5  # Allow more time - up to 5 seconds
            sleep_time = 1.0 / self.fps
        else:
            # Standard camera parameters
            max_tries = self.fps * 2  # Up to 2 seconds
            sleep_time = 1.0 / self.fps
            
        num_tries = 0
        while True:
            if self.color_image is not None:
                return self.color_image

            time.sleep(sleep_time)
            num_tries += 1
            
            # For ArduCam camera, allow more attempts and return a placeholder if needed
            if is_arducam:
                if num_tries > max_tries:
                    print(f"Warning: No frames received from ArduCam camera after {num_tries} attempts")
                    
                    if self.mock:
                        import tests.cameras.mock_cv2 as cv2
                    else:
                        import cv2
                    
                    # Create a placeholder image rather than failing
                    placeholder = np.zeros((self.capture_height, self.capture_width, 3), dtype=np.uint8)
                    
                    # Add red background
                    cv2.rectangle(placeholder, (0, 0), (self.capture_width, self.capture_height), 
                                 (0, 0, 60) if self.color_mode == "rgb" else (60, 0, 0), -1)
                    
                    # Add explanatory text
                    text_color = (0, 200, 255) if self.color_mode == "rgb" else (255, 200, 0)
                    messages = [
                        "ARDUCAM NOT RESPONDING",
                        "USB Camera unavailable",
                        "Check connections and restart"
                    ]
                    
                    y_pos = self.capture_height // 4
                    for msg in messages:
                        cv2.putText(
                            placeholder,
                            msg,
                            (self.capture_width // 10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            text_color,
                            2
                        )
                        y_pos += 40
                        
                    # Convert to RGB if needed
                    if self.color_mode == "rgb":
                        placeholder = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
                        
                    return placeholder
            elif num_tries > max_tries:
                raise TimeoutError(f"Timed out waiting for async_read() to start for camera {self.camera_index}.")

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"OpenCVCamera({self.camera_index}) is not connected. Try running `camera.connect()` first."
            )

        if self.thread is not None:
            self.stop_event.set()
            self.thread.join()  # wait for the thread to finish
            self.thread = None
            self.stop_event = None

        self.camera.release()
        self.camera = None
        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save a few frames using `OpenCVCamera` for all cameras connected to the computer, or a selected subset."
    )
    parser.add_argument(
        "--camera-ids",
        type=int,
        nargs="*",
        default=None,
        help="List of camera indices used to instantiate the `OpenCVCamera`. If not provided, find and use all available camera indices.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Set the number of frames recorded per seconds for all cameras. If not provided, use the default fps of each camera.",
    )
    parser.add_argument(
        "--width",
        type=str,
        default=None,
        help="Set the width for all cameras. If not provided, use the default width of each camera.",
    )
    parser.add_argument(
        "--height",
        type=str,
        default=None,
        help="Set the height for all cameras. If not provided, use the default height of each camera.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default="outputs/images_from_opencv_cameras",
        help="Set directory to save a few frames for each camera.",
    )
    parser.add_argument(
        "--record-time-s",
        type=float,
        default=4.0,
        help="Set the number of seconds used to record the frames. By default, 2 seconds.",
    )
    args = parser.parse_args()
    save_images_from_cameras(**vars(args))
