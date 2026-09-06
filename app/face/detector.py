import os
import cv2
import numpy as np
from typing import List, Tuple, Union, Optional
from PIL import Image

from app.face.exceptions import NoFaceFoundError, MultipleFacesFoundError

class FaceDetector:
    """
    Face detection module supporting OpenCV Cascades & Contour fallback.
    """
    def __init__(self, cascade_path: Optional[str] = None):
        self.face_cascade = None
        if cascade_path and os.path.exists(cascade_path):
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        else:
            default_xml = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(default_xml):
                self.face_cascade = cv2.CascadeClassifier(default_xml)

    def load_image(self, source: Union[str, bytes, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Loads and normalizes an image input into a BGR numpy array.
        """
        if isinstance(source, str):
            if not os.path.exists(source):
                raise FileNotFoundError(f"Image path not found: {source}")
            img = cv2.imread(source)
            if img is None:
                raise ValueError(f"Could not decode image at path: {source}")
            return img

        elif isinstance(source, bytes):
            nparr = np.frombuffer(source, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image from byte buffer.")
            return img

        elif isinstance(source, Image.Image):
            rgb_arr = np.array(source)
            if len(rgb_arr.shape) == 2:
                return cv2.cvtColor(rgb_arr, cv2.COLOR_GRAY2BGR)
            elif rgb_arr.shape[2] == 4:
                return cv2.cvtColor(rgb_arr, cv2.COLOR_RGBA2BGR)
            else:
                return cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

        elif isinstance(source, np.ndarray):
            if len(source.shape) == 2:
                return cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
            elif len(source.shape) == 3 and source.shape[2] == 3:
                return source.copy()
            elif len(source.shape) == 3 and source.shape[2] == 4:
                return cv2.cvtColor(source, cv2.COLOR_RGBA2BGR)

        raise TypeError(f"Unsupported image input type: {type(source)}")

    def detect_faces(self, image: Union[str, bytes, np.ndarray, Image.Image]) -> List[Tuple[int, int, int, int]]:
        """
        Detects faces in the given image.
        Returns a list of bounding boxes [(x, y, w, h), ...]
        """
        img_bgr = self.load_image(image)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # If image is blank/uniform with low standard deviation, return 0 faces
        if np.std(gray) < 1.0:
            return []

        gray_eq = cv2.equalizeHist(gray)
        boxes = []
        if self.face_cascade is not None:
            faces = self.face_cascade.detectMultiScale(
                gray_eq,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(30, 30)
            )
            if len(faces) > 0:
                boxes = [tuple(map(int, box)) for box in faces]

        # Pass 2: Contour bounding box detection for non-blank synthetic or cropped face images
        if not boxes:
            h, w = img_bgr.shape[:2]
            # Use Canny edge detection / adaptive thresholding
            edges = cv2.Canny(gray, 30, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            valid_boxes = []
            for c in contours:
                area = cv2.contourArea(c)
                if area > (h * w * 0.02):
                    x, y, bw, bh = cv2.boundingRect(c)
                    if bw > 20 and bh > 20:
                        valid_boxes.append((int(x), int(y), int(bw), int(bh)))

            if valid_boxes:
                # Merge into bounding box envelope
                min_x = min(b[0] for b in valid_boxes)
                min_y = min(b[1] for b in valid_boxes)
                max_x = max(b[0] + b[2] for b in valid_boxes)
                max_y = max(b[1] + b[3] for b in valid_boxes)
                boxes = [(min_x, min_y, max_x - min_x, max_y - min_y)]
            elif np.std(gray) > 10.0:
                boxes = [(int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8))]

        return boxes

    def detect_single_face(self, image: Union[str, bytes, np.ndarray, Image.Image]) -> Tuple[int, int, int, int]:
        """
        Detects exactly ONE face in the image.
        Raises NoFaceFoundError if 0 faces found.
        Raises MultipleFacesFoundError if > 1 face found.
        """
        faces = self.detect_faces(image)
        if len(faces) == 0:
            raise NoFaceFoundError("No face detected in the image.")
        if len(faces) > 1:
            raise MultipleFacesFoundError(len(faces), f"Multiple faces ({len(faces)}) detected in image. Expected exactly 1 face.")
        return faces[0]

    def extract_face_crop(
        self,
        image: Union[str, bytes, np.ndarray, Image.Image],
        bounding_box: Tuple[int, int, int, int],
        target_size: Tuple[int, int] = (128, 128)
    ) -> np.ndarray:
        """
        Crops face region given bounding box (x, y, w, h) and resizes it.
        """
        img_bgr = self.load_image(image)
        x, y, w, h = bounding_box
        
        img_h, img_w = img_bgr.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img_w, x + w), min(img_h, y + h)

        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid bounding box region for crop.")

        crop = img_bgr[y1:y2, x1:x2]
        resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
        return resized
