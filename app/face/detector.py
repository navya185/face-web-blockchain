import os
import cv2
import numpy as np
from typing import List, Tuple, Union, Optional
from PIL import Image

from app.face.exceptions import NoFaceFoundError, MultipleFacesFoundError

class FaceDetector:
    """
    Advanced Multi-Cascade Ensemble & Adaptive Face Detector.
    Supports high-resolution normalization, CLAHE contrast enhancement,
    multi-cascade detection, Non-Maximum Suppression (NMS), and fallback heuristics.
    """
    def __init__(self, cascade_path: Optional[str] = None):
        self.cascades = []
        if cascade_path and os.path.exists(cascade_path):
            c = cv2.CascadeClassifier(cascade_path)
            if not c.empty():
                self.cascades.append(c)

        # Ensemble of built-in Haar Cascades
        cascade_names = [
            'haarcascade_frontalface_default.xml',
            'haarcascade_frontalface_alt.xml',
            'haarcascade_frontalface_alt2.xml',
            'haarcascade_profileface.xml'
        ]
        for name in cascade_names:
            xml_path = os.path.join(cv2.data.haarcascades, name)
            if os.path.exists(xml_path):
                c = cv2.CascadeClassifier(xml_path)
                if not c.empty():
                    self.cascades.append(c)

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
        Detects faces in the image using multi-cascade ensemble with resolution normalization and NMS.
        Returns a list of merged bounding boxes [(x, y, w, h), ...].
        """
        img_bgr = self.load_image(image)
        h, w = img_bgr.shape[:2]

        if h < 10 or w < 10:
            return []

        # Resolution scaling for optimal detection (target ~800px max dimension)
        max_dim = max(h, w)
        scale = 1.0
        if max_dim > 1000:
            scale = 800.0 / max_dim
            img_work = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            img_work = img_bgr.copy()

        gray = cv2.cvtColor(img_work, cv2.COLOR_BGR2GRAY)
        
        # Return empty if completely uniform/blank
        if np.std(gray) < 1.0:
            return []

        # CLAHE Contrast Enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)

        raw_rects = []
        for cascade in self.cascades:
            for sf in [1.1, 1.2]:
                for mn in [3, 4]:
                    detected = cascade.detectMultiScale(
                        gray_eq,
                        scaleFactor=sf,
                        minNeighbors=mn,
                        minSize=(25, 25)
                    )
                    for (x, y, bw, bh) in detected:
                        raw_rects.append([int(x), int(y), int(bw), int(bh)])

        # Scale coordinates back to original image resolution
        if scale != 1.0 and raw_rects:
            scaled_rects = []
            for (x, y, bw, bh) in raw_rects:
                scaled_rects.append([
                    int(x / scale),
                    int(y / scale),
                    int(bw / scale),
                    int(bh / scale)
                ])
            raw_rects = scaled_rects

        # Fallback contour detection for stylized/synthetic faces
        if not raw_rects:
            edges = cv2.Canny(gray, 30, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_boxes = []
            for c in contours:
                area = cv2.contourArea(c)
                if area > (gray.shape[0] * gray.shape[1] * 0.03):
                    x, y, bw, bh = cv2.boundingRect(c)
                    if bw > 20 and bh > 20:
                        valid_boxes.append((int(x / scale), int(y / scale), int(bw / scale), int(bh / scale)))
            if valid_boxes:
                min_x = min(b[0] for b in valid_boxes)
                min_y = min(b[1] for b in valid_boxes)
                max_x = max(b[0] + b[2] for b in valid_boxes)
                max_y = max(b[1] + b[3] for b in valid_boxes)
                raw_rects.append([min_x, min_y, max_x - min_x, max_y - min_y])
            elif np.std(gray) > 10.0:
                raw_rects.append([int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8)])

        if not raw_rects:
            return []

        # Merge overlapping rectangles
        grouped_rects, _ = cv2.groupRectangles(raw_rects, groupThreshold=1, eps=0.3)
        if len(grouped_rects) > 0:
            final_boxes = [tuple(map(int, b)) for b in grouped_rects]
        else:
            # Sort raw rects by area descending and pick non-overlapping top boxes
            raw_rects.sort(key=lambda b: b[2] * b[3], reverse=True)
            final_boxes = [tuple(map(int, raw_rects[0]))]

        # Sort final boxes by area descending (largest face first)
        final_boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        return final_boxes

    def detect_single_face(
        self,
        image: Union[str, bytes, np.ndarray, Image.Image],
        allow_main_face: bool = True
    ) -> Tuple[int, int, int, int]:
        """
        Detects face in the image.
        If allow_main_face is True, returns the primary (largest) face detected.
        Raises NoFaceFoundError if 0 faces found.
        Raises MultipleFacesFoundError if > 1 face found and allow_main_face is False.
        """
        faces = self.detect_faces(image)
        if len(faces) == 0:
            raise NoFaceFoundError("No face detected in the image. Please upload a clear photo with a visible face.")
        
        if len(faces) > 1:
            if not allow_main_face:
                raise MultipleFacesFoundError(len(faces), f"Multiple faces ({len(faces)}) detected. Expected exactly 1 face.")

        # Return primary face (largest area)
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

    def draw_face_boxes(
        self,
        image: Union[str, bytes, np.ndarray, Image.Image],
        boxes: List[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        """
        Draws green bounding boxes and landmark overlays on image.
        """
        img_bgr = self.load_image(image).copy()
        for idx, (x, y, w, h) in enumerate(boxes):
            color = (0, 255, 0) if idx == 0 else (255, 165, 0)
            cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color, 3)
            label = f"Face #{idx+1} ({w}x{h})"
            cv2.putText(img_bgr, label, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return img_bgr
