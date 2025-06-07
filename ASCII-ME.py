import cv2
import numpy as np
import time
import sys
import argparse
from typing import Tuple, List


class ASCIICamera:
    def __init__(self):
        self.ascii_chars = "@%#*+=-:. "  # Default ASCII gradient
        self.colored = False
        self.invert = False
        self.font_ratio = 0.43  # Width to height ratio of font characters
        self.target_width = 100
        self.blur = False
        self.edge_detection = False
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.capture = cv2.VideoCapture(0)

        if not self.capture.isOpened():
            print("Error: Could not open webcam.")
            sys.exit(1)

        # Set a lower resolution for better performance
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def update_fps(self):
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 1:  # Update FPS every second
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()

    def get_ascii_char(self, pixel_value: int) -> str:
        """Map pixel value to ASCII character"""
        if self.invert:
            pixel_value = 255 - pixel_value

        index = int(pixel_value / 255 * (len(self.ascii_chars) - 1))
        return self.ascii_chars[index]

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target dimensions maintaining aspect ratio"""
        height, width = image.shape[:2]
        target_height = int(self.target_width * (height / width) * self.font_ratio)
        return cv2.resize(image, (self.target_width, target_height))

    def apply_effects(self, image: np.ndarray) -> np.ndarray:
        """Apply image processing effects"""
        if self.blur:
            image = cv2.GaussianBlur(image, (3, 3), 0)

        if self.edge_detection:
            image = cv2.Canny(image, 100, 200)

        return image

    def convert_frame_to_ascii(self, frame: np.ndarray) -> str:
        """Convert a frame to ASCII art"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply effects
        gray = self.apply_effects(gray)

        # Resize
        resized = self.resize_image(gray)

        # Convert to ASCII
        ascii_art = []
        for row in resized:
            ascii_row = [self.get_ascii_char(pixel) for pixel in row]
            ascii_art.append("".join(ascii_row))

        return "\n".join(ascii_art)

    def display_controls(self):
        """Display control instructions"""
        controls = [
            "Controls:",
            "  [C] Toggle color",
            "  [I] Invert brightness",
            "  [B] Toggle blur",
            "  [E] Toggle edge detection",
            "  [+] Increase width",
            "  [-] Decrease width",
            "  [Q] Quit"
        ]
        print("\n" + "\n".join(controls) + "\n")

    def run(self):
        self.display_controls()

        try:
            while True:
                ret, frame = self.capture.read()
                if not ret:
                    break

                self.update_fps()

                # Convert frame to ASCII
                ascii_frame = self.convert_frame_to_ascii(frame)

                # Clear screen and print ASCII frame
                print("\033[H\033[J")  # Clear terminal
                print(f"ASCII Camera - FPS: {self.fps:.1f}")
                print(ascii_frame)

                # Check for key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.colored = not self.colored
                elif key == ord('i'):
                    self.invert = not self.invert
                elif key == ord('b'):
                    self.blur = not self.blur
                elif key == ord('e'):
                    self.edge_detection = not self.edge_detection
                elif key == ord('+'):
                    self.target_width = min(self.target_width + 10, 200)
                elif key == ord('-'):
                    self.target_width = max(self.target_width - 10, 40)

        finally:
            self.capture.release()
            cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="ASCII Camera Effect")
    parser.add_argument('--width', type=int, default=100, help="Initial ASCII art width")
    parser.add_argument('--invert', action='store_true', help="Start with inverted colors")
    parser.add_argument('--blur', action='store_true', help="Start with blur effect")
    parser.add_argument('--edges', action='store_true', help="Start with edge detection")
    parser.add_argument('--chars', type=str, default="@%#*+=-:. ", help="Custom ASCII gradient")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    ascii_cam = ASCIICamera()
    ascii_cam.target_width = args.width
    ascii_cam.invert = args.invert
    ascii_cam.blur = args.blur
    ascii_cam.edge_detection = args.edges
    ascii_cam.ascii_chars = args.chars

    ascii_cam.run()