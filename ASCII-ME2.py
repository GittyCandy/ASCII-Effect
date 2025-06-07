import cv2
import numpy as np
import time
import sys
import argparse
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional


class ColorMode(Enum):
    MONOCHROME = 1
    GRAYSCALE = 2
    COLOR = 3


@dataclass
class AsciiArtConfig:
    cols: int = 120
    scale: float = 0.43
    color_mode: ColorMode = ColorMode.COLOR
    brightness: float = 1.0
    contrast: float = 1.0
    ascii_chars: str = "@%#*+=-:. "
    live_preview: bool = True
    fps: bool = True
    invert: bool = False


class AsciiCamera:
    def __init__(self, config: AsciiArtConfig):
        self.config = config
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video device")

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.prev_frame_time = 0
        self.new_frame_time = 0
        self.running = True

        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def apply_brightness_contrast(self, img: np.ndarray) -> np.ndarray:
        """Apply brightness and contrast adjustments to the image"""
        img = img.astype('float32')
        img = (img - 127.5) * self.config.contrast + 127.5
        img = img * self.config.brightness
        return np.clip(img, 0, 255).astype('uint8')

    def pixel_to_ascii(self, pixel_value: int) -> str:
        """Map a pixel value (0-255) to an ASCII char"""
        if self.config.invert:
            pixel_value = 255 - pixel_value
        return self.config.ascii_chars[pixel_value * len(self.config.ascii_chars) // 256]

    def frame_to_ascii(self, frame: np.ndarray) -> Tuple[List[str], Optional[np.ndarray]]:
        """Convert a frame to ASCII art with optional color mapping"""
        # Apply brightness/contrast adjustments
        frame = self.apply_brightness_contrast(frame)

        # Convert to grayscale for ASCII mapping
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        # Calculate cell dimensions
        cell_width = width // self.config.cols
        cell_height = int(cell_width / self.config.scale)
        rows = height // cell_height

        ascii_image = []
        color_map = None

        if self.config.color_mode == ColorMode.COLOR:
            color_map = np.zeros((rows, self.config.cols, 3), dtype=np.uint8)

        for i in range(rows):
            y1 = i * cell_height
            y2 = min((i + 1) * cell_height, height)

            ascii_row = []
            for j in range(self.config.cols):
                x1 = j * cell_width
                x2 = min((j + 1) * cell_width, width)

                # Get region of interest
                roi_gray = gray[y1:y2, x1:x2]
                avg_gray = int(np.mean(roi_gray))

                # Get ASCII character
                char = self.pixel_to_ascii(avg_gray)
                ascii_row.append(char)

                # Store color information if needed
                if self.config.color_mode == ColorMode.COLOR:
                    roi_color = frame[y1:y2, x1:x2]
                    avg_color = np.mean(roi_color, axis=(0, 1))
                    color_map[i, j] = avg_color.astype(int)

            ascii_image.append("".join(ascii_row))

        return ascii_image, color_map

    def render_ascii_art(self, ascii_image: List[str], color_map: Optional[np.ndarray] = None) -> np.ndarray:
        """Render ASCII art to an image with optional coloring"""
        rows = len(ascii_image)
        cols = len(ascii_image[0]) if rows > 0 else 0

        # Calculate image dimensions (10px per row, 8px per column)
        img_height = rows * 10 + 40  # Extra space for UI elements
        img_width = cols * 8
        img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

        # Draw ASCII characters
        for i, row in enumerate(ascii_image):
            for j, c in enumerate(row):
                color = (255, 255, 255)
                if color_map is not None:
                    color = tuple(map(int, color_map[i, j]))
                elif self.config.color_mode == ColorMode.GRAYSCALE:
                    gray_value = 255 - (self.config.ascii_chars.index(c) * 255 // len(self.config.ascii_chars))
                    color = (gray_value, gray_value, gray_value)

                cv2.putText(img, c, (j * 8, i * 10 + 10),
                            self.font, 0.4, color, 1, cv2.LINE_AA)

        # Add UI elements
        self._draw_ui(img, rows, cols)

        return img

    def _draw_ui(self, img: np.ndarray, rows: int, cols: int) -> None:
        """Draw UI elements on the image"""
        # Draw header
        header_text = "ASCII Camera - Press 'Q' to quit"
        cv2.putText(img, header_text, (10, rows * 10 + 20),
                    self.font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw mode information
        mode_text = f"Mode: {self.config.color_mode.name} | Chars: {len(self.config.ascii_chars)} | Size: {cols}x{rows}"
        cv2.putText(img, mode_text, (10, rows * 10 + 35),
                    self.font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        # Draw FPS if enabled
        if self.config.fps:
            self.new_frame_time = time.time()
            fps = 1 / (self.new_frame_time - self.prev_frame_time)
            self.prev_frame_time = self.new_frame_time
            fps_text = f"FPS: {int(fps)}"
            cv2.putText(img, fps_text, (img.shape[1] - 80, 20),
                        self.font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    def handle_keypress(self, key: int) -> None:
        """Handle keyboard input"""
        # Toggle color modes
        if key == ord('c'):
            modes = list(ColorMode)
            current_idx = modes.index(self.config.color_mode)
            self.config.color_mode = modes[(current_idx + 1) % len(modes)]

        # Adjust brightness
        elif key == ord('+'):
            self.config.brightness = min(self.config.brightness + 0.1, 2.0)
        elif key == ord('-'):
            self.config.brightness = max(self.config.brightness - 0.1, 0.1)

        # Adjust contrast
        elif key == ord('>'):
            self.config.contrast = min(self.config.contrast + 0.1, 2.0)
        elif key == ord('<'):
            self.config.contrast = max(self.config.contrast - 0.1, 0.1)

        # Adjust columns/resolution
        elif key == ord(']'):
            self.config.cols = min(self.config.cols + 10, 300)
        elif key == ord('['):
            self.config.cols = max(self.config.cols - 10, 40)

        # Toggle inversion
        elif key == ord('i'):
            self.config.invert = not self.config.invert

        # Toggle FPS display
        elif key == ord('f'):
            self.config.fps = not self.config.fps

        # Quit program
        elif key == ord('q'):
            self.running = False

    def run(self) -> None:
        """Main application loop"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Process frame
            ascii_image, color_map = self.frame_to_ascii(frame)

            if self.config.live_preview:
                # Show original and ASCII side by side
                ascii_art = self.render_ascii_art(ascii_image,
                                                  color_map if self.config.color_mode == ColorMode.COLOR else None)

                # Resize original for display
                original_resized = cv2.resize(frame, (ascii_art.shape[1], ascii_art.shape[0]))

                # Combine images
                combined = np.hstack((original_resized, ascii_art))
                cv2.imshow('ASCII Camera - Original (Left) | ASCII (Right)', combined)
            else:
                # Show only ASCII
                ascii_art = self.render_ascii_art(ascii_image,
                                                  color_map if self.config.color_mode == ColorMode.COLOR else None)
                cv2.imshow('ASCII Camera', ascii_art)

            # Handle keypress
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                self.handle_keypress(key)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Advanced ASCII Camera')
    parser.add_argument('--cols', type=int, default=120, help='Number of ASCII columns')
    parser.add_argument('--scale', type=float, default=0.43, help='Character aspect ratio scale')
    parser.add_argument('--chars', type=str, default="@%#*+=-:. ", help='ASCII characters to use')
    parser.add_argument('--mono', action='store_true', help='Monochrome mode')
    parser.add_argument('--gray', action='store_true', help='Grayscale mode')
    parser.add_argument('--color', action='store_true', help='Color mode (default)')
    parser.add_argument('--no-preview', action='store_true', help='Disable live preview')
    parser.add_argument('--no-fps', action='store_true', help='Disable FPS counter')
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine color mode
    if args.mono:
        color_mode = ColorMode.MONOCHROME
    elif args.gray:
        color_mode = ColorMode.GRAYSCALE
    else:
        color_mode = ColorMode.COLOR

    # Create config
    config = AsciiArtConfig(
        cols=args.cols,
        scale=args.scale,
        color_mode=color_mode,
        ascii_chars=args.chars,
        live_preview=not args.no_preview,
        fps=not args.no_fps
    )

    # Run application
    try:
        app = AsciiCamera(config)
        app.run()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()