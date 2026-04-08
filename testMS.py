#!/usr/bin/env python3
"""
Camera test script that properly handles both:
- Phone camera (loopback device) at /dev/video4
- Arducam USB Camera (gripper) at /dev/video0
"""
import cv2
import os
import time
import numpy as np
import threading
import queue
import subprocess
import signal

# Camera settings
PHONE_DEVICE = "/dev/video4"   # Phone camera (loopback device)
GRIPPER_DEVICE = "/dev/video0" # Arducam USB Camera
WIDTH = 640
HEIGHT = 480
FPS = 30

# Debug and configuration settings
DEBUG = True
ARDUCAM_TIMEOUT = 0.5  # Timeout for Arducam frame read in seconds

# Simplified timeout handler
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def with_timeout(seconds, func, *args, **kwargs):
    """Run function with a timeout"""
    # Make sure any previous alarm is canceled
    signal.alarm(0)
    
    # Set the timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(seconds))
    try:
        result = func(*args, **kwargs)
        signal.alarm(0)  # Cancel the alarm immediately after function returns
        return result
    except TimeoutError as e:
        print(f"Timeout: {e}")
        return False
    except Exception as e:
        print(f"Exception during timeout operation: {e}")
        return False
    finally:
        # Cancel the alarm
        signal.alarm(0)

def log(message):
    """Print log message if debug is enabled"""
    if DEBUG:
        print(f"[DEBUG] {message}")

def get_device_info():
    """Get information about video devices"""
    try:
        # Run v4l2-ctl to get device information
        result = subprocess.run(['v4l2-ctl', '--list-devices'], 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE, 
                            text=True,
                            check=False)
        if result.returncode == 0:
            print("\n=== Connected Camera Devices ===")
            print(result.stdout)
            return True
        else:
            print("Error listing devices:", result.stderr)
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

class CameraReader:
    """Thread-safe camera reader class that handles timeouts"""
    
    def __init__(self, device_path, width, height, fps, timeout=1.0, name="Camera"):
        """Initialize the camera reader"""
        self.device_path = device_path
        self.width = width
        self.height = height
        self.fps = fps
        self.timeout = timeout
        self.name = name
        
        self.frame_queue = queue.Queue(maxsize=1)  # Only keep latest frame
        self.running = False
        self.cap = None
        self.thread = None
        self.last_frame_time = 0
        self.frame_count = 0
        self.start_time = 0
        self.color_image = None
        self.logs = {"delta_timestamp_s": 0.0}
        
    def setup_camera(self):
        """Set up the camera with appropriate settings"""
        try:
            # Open the camera
            log(f"Opening {self.name} at {self.device_path}")
            self.cap = cv2.VideoCapture(self.device_path)
            
            if not self.cap.isOpened():
                log(f"Failed to open {self.name}")
                return False
            
            # Configure camera
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Apply specific settings for Arducam with timeout
            if "Arducam" in self.name or self.device_path == GRIPPER_DEVICE:
                log("Applying Arducam-specific settings")
                try:
                    # Set alarm for timeout
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(int(self.timeout))
                    
                    # Try to apply settings with timeout
                    self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)   # Default 0, range -64 to 64
                    self.cap.set(cv2.CAP_PROP_CONTRAST, 32)    # Default 32, range 0 to 64
                    self.cap.set(cv2.CAP_PROP_SATURATION, 64)  # Default 64, range 0 to 128
                    self.cap.set(cv2.CAP_PROP_GAIN, 0)         # Default 0, range 0 to 100
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # Minimize buffering
                    
                    # Try to set pixel format to MJPG for better performance
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                    
                    # Cancel the alarm
                    signal.alarm(0)
                except TimeoutError:
                    # If we time out, continue anyway
                    log(f"WARNING: Arducam settings timed out after {self.timeout}s, continuing anyway")
                    signal.alarm(0)
                except Exception as e:
                    # If other error, log and continue
                    log(f"Error in Arducam settings: {e}")
                    signal.alarm(0)
            
            # Test the connection with a read, with timeout
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(self.timeout))
                
                ret, test_frame = self.cap.read()
                
                signal.alarm(0)
                
                if not ret:
                    log(f"Failed initial read from {self.name}")
                    self.cap.release()
                    return False
            except TimeoutError:
                log(f"Initial frame read timed out for {self.name}")
                self.cap.release()
                return False
            except Exception as e:
                log(f"Error during initial frame read: {e}")
                self.cap.release()
                return False
            
            # Get actual camera settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            try:
                fourcc_int = int(self.cap.get(cv2.CAP_PROP_FOURCC))
                fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
            except:
                fourcc_str = "Unknown"
            
            print(f"{self.name} ready:")
            print(f"  Resolution: {actual_width}x{actual_height}")
            print(f"  FPS: {actual_fps}")
            print(f"  Format: {fourcc_str}")
            
            return True
            
        except Exception as e:
            print(f"Error setting up {self.name}: {e}")
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
            return False
    
    def start(self):
        """Start the camera reader thread"""
        if self.running:
            return True
            
        if not self.setup_camera():
            return False
            
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._reader_thread, daemon=True)
        self.thread.start()
        return True
    
    def _reader_thread(self):
        """Thread that continuously reads frames from the camera"""
        while self.running and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                
                if ret:
                    # Successfully got a frame, update stats
                    self.frame_count += 1
                    self.last_frame_time = time.time()
                    
                    # Replace old frame with new one
                    try:
                        # Clear the queue of old frames
                        while not self.frame_queue.empty():
                            self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    
                    # Add the new frame
                    self.frame_queue.put((ret, frame))
                    self.color_image = frame
                else:
                    log(f"{self.name} read returned no frame")
                    time.sleep(0.01)  # Short pause to prevent busy waiting
            
            except Exception as e:
                log(f"Error in {self.name} reader thread: {e}")
                time.sleep(0.1)  # Pause on error
    
    def get_frame(self):
        """Get the latest frame with timeout protection"""
        if not self.running:
            return False, None
            
        try:
            # Try to get a frame from the queue with timeout
            return self.frame_queue.get(timeout=self.timeout)
        except queue.Empty:
            # If queue is empty after timeout, no new frames are coming
            current_time = time.time()
            time_since_last = current_time - self.last_frame_time
            
            if time_since_last > self.timeout * 2:
                log(f"{self.name} hasn't received frames for {time_since_last:.1f}s")
            
            return False, None
    
    def read(self):
        """Get the latest frame converted to RGB"""
        ret, frame = self.get_frame()
        start_time = time.time()
        
        if ret and frame is not None:
            # For LeRobot compatibility - convert to RGB and return
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.logs["delta_timestamp_s"] = time.time() - start_time
            return frame_rgb
        else:
            # If we have a cached frame, return it
            if self.color_image is not None:
                frame_rgb = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
                self.logs["delta_timestamp_s"] = time.time() - start_time
                return frame_rgb
                
            # Create a blank frame as a fallback
            blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            self.logs["delta_timestamp_s"] = time.time() - start_time
            return blank
    
    def async_read(self):
        """Non-blocking read for LeRobot compatibility"""
        return self.read()
        
    def get_fps(self):
        """Calculate current FPS"""
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            return self.frame_count / elapsed
        return 0
    
    def stop(self):
        """Stop the camera reader thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Clear queue
        try:
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait()
        except:
            pass
            
    def disconnect(self):
        """Alias for stop() for LeRobot compatibility"""
        self.stop()
        
    def connect(self):
        """Alias for start() for LeRobot compatibility"""
        return self.start()

def enhance_dark_image(frame):
    """Enhance a dark image to make it more visible"""
    if frame is None:
        return None
    
    # Check brightness
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    
    # If image is too dark, enhance it
    if brightness < 50:
        # Increase brightness
        alpha = 2.0  # Contrast control
        beta = 50    # Brightness control
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        
        # Apply CLAHE for better contrast in dark areas
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return frame

def main():
    """Main function to display the phone and gripper camera streams"""
    # Check if display is available
    if "DISPLAY" not in os.environ:
        print("ERROR: No display available. Set with: export DISPLAY=:0")
        return False
        
    print(f"Display found: {os.environ['DISPLAY']}")
    print(f"OpenCV version: {cv2.__version__}")
    
    # Get information about devices
    get_device_info()
    
    # Create windows for displaying video
    cv2.namedWindow("Phone Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Phone Camera", WIDTH, HEIGHT)
    cv2.moveWindow("Phone Camera", 50, 100)
    
    cv2.namedWindow("Gripper Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gripper Camera", WIDTH, HEIGHT)
    cv2.moveWindow("Gripper Camera", WIDTH + 100, 100)
    
    # Create camera readers with appropriate timeouts
    phone_camera = CameraReader(
        PHONE_DEVICE, 
        WIDTH, 
        HEIGHT, 
        FPS, 
        timeout=0.5,
        name="Phone Camera"
    )
    
    gripper_camera = CameraReader(
        GRIPPER_DEVICE, 
        WIDTH, 
        HEIGHT, 
        FPS, 
        timeout=ARDUCAM_TIMEOUT,
        name="Arducam Gripper"
    )
    
    # Start phone camera (the reliable one)
    if not phone_camera.start():
        print("Failed to start phone camera")
        return False
    
    # Start gripper camera (may not work)
    gripper_working = gripper_camera.start()
    if not gripper_working:
        print("WARNING: Failed to start gripper camera")
    
    # Create a placeholder for when gripper camera doesn't work
    gripper_placeholder = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    
    # Red background
    cv2.rectangle(gripper_placeholder, (0, 0), (WIDTH, HEIGHT), (0, 0, 60), -1)
    
    # Add explanatory text
    messages = [
        "GRIPPER CAMERA UNAVAILABLE",
        "Arducam USB Camera not responding",
        "Check USB connections and try again",
        "",
        "Press 'r' to retry connection",
        "Press 'e' to toggle enhancement",
        "Press 'q' to quit"
    ]
    
    y_pos = HEIGHT // 4
    for msg in messages:
        cv2.putText(
            gripper_placeholder,
            msg,
            (WIDTH // 10, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 255),
            2
        )
        y_pos += 40
    
    print("\nCamera Test Script")
    print("=================")
    print("Press 'r' to retry gripper camera")
    print("Press 'e' to toggle image enhancement")
    print("Press 'q' or ESC to exit")
    
    # Main loop
    running = True
    enhance_mode = True  # Start with enhancement on
    
    while running:
        # Get frame from phone camera
        ret_phone, phone_frame = phone_camera.get_frame()
        
        if ret_phone:
            # Add text overlay
            timestamp = time.strftime("%H:%M:%S")
            cv2.putText(
                phone_frame,
                f"Phone - {timestamp} - {phone_camera.get_fps():.1f} FPS",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Show phone frame
            cv2.imshow("Phone Camera", phone_frame)
        
        # Get frame from gripper camera
        ret_gripper, gripper_frame = gripper_camera.get_frame()
        
        if ret_gripper and gripper_frame is not None:
            # Apply enhancement if needed
            if enhance_mode:
                gripper_frame = enhance_dark_image(gripper_frame)
            
            # Add text overlay
            timestamp = time.strftime("%H:%M:%S")
            cv2.putText(
                gripper_frame,
                f"Gripper - {timestamp} - {gripper_camera.get_fps():.1f} FPS - {'Enhanced' if enhance_mode else 'Normal'}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Show gripper frame
            cv2.imshow("Gripper Camera", gripper_frame)
            gripper_working = True
        else:
            # Show placeholder if gripper camera isn't working
            cv2.imshow("Gripper Camera", gripper_placeholder)
            if gripper_working:
                print("Gripper camera stopped responding")
                gripper_working = False
        
        # Check for key press (ESC=27, q=113, r=114, e=101)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or 'q'
            print("User requested exit")
            running = False
        elif key == ord('r'):  # 'r' to retry gripper
            print("Retrying gripper camera connection...")
            gripper_camera.stop()
            time.sleep(0.5)  # Short pause before reconnecting
            gripper_working = gripper_camera.start()
        elif key == ord('e'):  # 'e' to toggle enhancement
            enhance_mode = not enhance_mode
            print(f"Image enhancement: {'ON' if enhance_mode else 'OFF'}")
    
    # Clean up
    print("Cleaning up resources...")
    phone_camera.stop()
    gripper_camera.stop()
    
    cv2.destroyAllWindows()
    
    # Force close windows (sometimes needed in Windows/WSL)
    for _ in range(5):
        cv2.waitKey(1)
    
    print("Camera closed and resources released")
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Make sure all windows are closed on exit
        cv2.destroyAllWindows() 