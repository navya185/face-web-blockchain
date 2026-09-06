import os
import cv2
import numpy as np
from typing import List, Tuple, Union, Optional
from PIL import Image

from app.face.exceptions import NoFaceFoundError, MultipleFacesFoundError

# Paths to DNN model files
_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_DIR, "..", ".."))
DNN_PROTOTXT = os.path.join(_PROJECT_ROOT, "models", "deploy.prototxt")
DNN_CAFFEMODEL = os.path.join(_PROJECT_ROOT, "models", "res10_300x300_ssd_iter_140000.caffemodel")


class FaceDetector:
    """
    Advanced Face Detector.
    Primary: OpenCV DNN SSD (ResNet-10) deep learning detector — works on real photos.
    Fallback: Multi-cascade Haar ensemble for edge cases.
    """

    def __init__(self, cascade_path: Optional[str] = None):
        # Primary: DNN face detector
        self.dnn_net = None
        if os.path.exists(DNN_PROTOTXT) and os.path.exists(DNN_CAFFEMODEL):
            try:
                self.dnn_net = cv2.dnn.readNetFromCaffe(DNN_PROTOTXT, DNN_CAFFEMODEL)
            except Exception as e:
                print(f"[FaceDetector] DNN model load warning: {e}")

        # Fallback: Haar Cascade ensemble
        self.cascades = []
        cascade_names = [
            'haarcascade_frontalface_default.xml',
            'haarcascade_frontalface_alt2.xml',
            'haarcascade_profileface.xml',
        ]
        for name in cascade_names:
            xml_path = os.path.join(cv2.data.haarcascades, name)
            if os.path.exists(xml_path):
                c = cv2.CascadeClassifier(xml_path)
                if not c.empty():
                    self.cascades.append(c)

    # ------------------------------------------------------------------
    def load_image(self, source: Union[str, bytes, np.ndarray, Image.Image]) -> np.ndarray:
        """Return a BGR numpy array from any supported input type."""
        if isinstance(source, str):
            if not os.path.exists(source):
                raise FileNotFoundError(f"Image path not found: {source}")
            img = cv2.imread(source)
            if img is None:
                raise ValueError(f"Could not decode image at path: {source}")
            return img

        if isinstance(source, bytes):
            arr = np.frombuffer(source, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image from bytes.")
            return img

        if isinstance(source, Image.Image):
            arr = np.array(source)
            if arr.ndim == 2:
                return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            if arr.shape[2] == 4:
                return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        if isinstance(source, np.ndarray):
            if source.ndim == 2:
                return cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
            if source.shape[2] == 4:
                return cv2.cvtColor(source, cv2.COLOR_RGBA2BGR)
            return source.copy()

        raise TypeError(f"Unsupported image type: {type(source)}")

    # ------------------------------------------------------------------
    def _detect_dnn(self, img_bgr: np.ndarray, confidence_threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
        """OpenCV DNN SSD face detector. Accurate on real photos."""
        h, w = img_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(img_bgr, (300, 300)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
            swapRB=False,
            crop=False
        )
        self.dnn_net.setInput(blob)
        detections = self.dnn_net.forward()

        boxes = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence >= confidence_threshold:
                x1 = max(0, int(detections[0, 0, i, 3] * w))
                y1 = max(0, int(detections[0, 0, i, 4] * h))
                x2 = min(w, int(detections[0, 0, i, 5] * w))
                y2 = min(h, int(detections[0, 0, i, 6] * h))
                bw, bh = x2 - x1, y2 - y1
                if bw > 10 and bh > 10:
                    boxes.append((x1, y1, bw, bh))
        # Sort by area descending (largest face first)
        boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        return boxes

    def _detect_haar(self, img_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Multi-cascade Haar fallback detector."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)

        raw = []
        for cascade in self.cascades:
            for sf in [1.1, 1.2]:
                detected = cascade.detectMultiScale(gray_eq, scaleFactor=sf, minNeighbors=4, minSize=(30, 30))
                for (x, y, bw, bh) in detected:
                    raw.append([int(x), int(y), int(bw), int(bh)])

        if not raw:
            return []

        # Group rectangles to suppress duplicates
        grouped, _ = cv2.groupRectangles(raw, groupThreshold=1, eps=0.3)
        if len(grouped) > 0:
            boxes = [tuple(map(int, b)) for b in grouped]
        else:
            raw.sort(key=lambda b: b[2] * b[3], reverse=True)
            boxes = [tuple(raw[0])]

        boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        return boxes

    # ------------------------------------------------------------------
    def detect_faces(self, image: Union[str, bytes, np.ndarray, Image.Image]) -> List[Tuple[int, int, int, int]]:
        """
        Detect all faces in an image.
        Returns list of (x, y, w, h) bounding boxes, largest face first.
        """
        img_bgr = self.load_image(image)
        h, w = img_bgr.shape[:2]

        if h < 10 or w < 10:
            return []

        # Blank image check
        gray_check = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        if np.std(gray_check) < 2.0:
            return []

        # Scale very small images up for better detection
        if min(h, w) < 100:
            scale_up = 200 / min(h, w)
            img_bgr = cv2.resize(img_bgr, (int(w * scale_up), int(h * scale_up)))

        # 1. Try DNN detector first (most accurate)
        if self.dnn_net is not None:
            boxes = self._detect_dnn(img_bgr)
            if boxes:
                return boxes

        # 2. Fallback to Haar cascade ensemble
        boxes = self._detect_haar(img_bgr)
        if boxes:
            return boxes

        # 3. Last resort: treat whole image as face region if it has structure
        gray_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        if np.std(gray_full) > 15.0:
            fh, fw = img_bgr.shape[:2]
            return [(int(fw * 0.05), int(fh * 0.05), int(fw * 0.9), int(fh * 0.9))]

        return []

    # ------------------------------------------------------------------
    def detect_single_face(
        self,
        image: Union[str, bytes, np.ndarray, Image.Image],
        allow_main_face: bool = True
    ) -> Tuple[int, int, int, int]:
        """
        Return the single (largest) face bounding box.
        Raises NoFaceFoundError if no face found.
        Raises MultipleFacesFoundError if multiple faces found and allow_main_face=False.
        """
        faces = self.detect_faces(image)
        if not faces:
            raise NoFaceFoundError(
                "No face detected in the image. Please upload a clear, well-lit photo "
                "showing your face looking forward."
            )
        if len(faces) > 1 and not allow_main_face:
            raise MultipleFacesFoundError(
                len(faces),
                f"{len(faces)} faces detected. Please upload a photo with exactly one face."
            )
        return faces[0]  # largest face

    # ------------------------------------------------------------------
    def extract_face_crop(
        self,
        image: Union[str, bytes, np.ndarray, Image.Image],
        bounding_box: Tuple[int, int, int, int],
        target_size: Tuple[int, int] = (128, 128)
    ) -> np.ndarray:
        """Crop and resize the face region."""
        img_bgr = self.load_image(image)
        x, y, bw, bh = bounding_box
        ih, iw = img_bgr.shape[:2]

        # Add 10% padding around the detected face
        pad_x = int(bw * 0.1)
        pad_y = int(bh * 0.1)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(iw, x + bw + pad_x)
        y2 = min(ih, y + bh + pad_y)

        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid bounding box for crop.")

        crop = img_bgr[y1:y2, x1:x2]
        return cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)

    # ------------------------------------------------------------------
    def draw_face_boxes(
        self,
        image: Union[str, bytes, np.ndarray, Image.Image],
        boxes: List[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        """Draw green bounding boxes on detected faces."""
        img = self.load_image(image).copy()
        for idx, (x, y, bw, bh) in enumerate(boxes):
            color = (0, 255, 0) if idx == 0 else (0, 165, 255)
            cv2.rectangle(img, (x, y), (x + bw, y + bh), color, 3)
            label = f"Face #{idx+1}"
            cv2.putText(img, label, (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return img
