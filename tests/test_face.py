import os
import cv2
import numpy as np
import pytest

from app.face.detector import FaceDetector
from app.face.embedder import FaceEmbedder
from app.face.exceptions import NoFaceFoundError, MultipleFacesFoundError

@pytest.fixture
def sample_face_image(tmp_path):
    img_path = str(tmp_path / "single_face.jpg")
    img = np.full((300, 300, 3), (240, 240, 240), dtype=np.uint8)
    
    # Draw face with distinct color skin (BGR: 150, 180, 220)
    cv2.ellipse(img, (150, 140), (80, 110), 0, 0, 360, (150, 180, 220), -1)
    cv2.circle(img, (120, 110), 12, (255, 255, 255), -1)
    cv2.circle(img, (180, 110), 12, (255, 255, 255), -1)
    cv2.circle(img, (120, 110), 5, (0, 0, 0), -1)
    cv2.circle(img, (180, 110), 5, (0, 0, 0), -1)
    cv2.ellipse(img, (150, 180), (25, 10), 0, 0, 180, (50, 50, 180), 3)

    cv2.imwrite(img_path, img)
    return img_path

@pytest.fixture
def blank_image(tmp_path):
    img_path = str(tmp_path / "blank.jpg")
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(img_path, img)
    return img_path

def test_face_detector_single_face(sample_face_image):
    detector = FaceDetector()
    box = detector.detect_single_face(sample_face_image)
    assert len(box) == 4
    assert box[2] > 0 and box[3] > 0

def test_face_detector_no_face_error(blank_image):
    detector = FaceDetector()
    with pytest.raises(NoFaceFoundError):
        detector.detect_single_face(blank_image)

def test_face_embedder_encoding(sample_face_image):
    embedder = FaceEmbedder()
    embedding, box = embedder.encode_face(sample_face_image)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (128,)
    assert len(box) == 4

def test_face_similarity_metrics():
    vec1 = np.ones(128, dtype=np.float32) / np.sqrt(128)
    vec2 = np.ones(128, dtype=np.float32) / np.sqrt(128)
    sim = FaceEmbedder.calculate_similarity(vec1, vec2)
    dist = FaceEmbedder.calculate_distance(vec1, vec2)
    assert pytest.approx(sim, 0.001) == 1.0
    assert pytest.approx(dist, 0.001) == 0.0
