import os
import cv2
import numpy as np

def generate_sample_face_image(output_path: str, name: str = "Test Subject"):
    """
    Generates a synthetic portrait image with detectable facial landmarks
    for reliable local testing without requiring external photo assets.
    """
    # Create 300x300 canvas
    img = np.full((300, 300, 3), (240, 240, 240), dtype=np.uint8)

    # Head oval
    cv2.ellipse(img, (150, 140), (80, 110), 0, 0, 360, (210, 180, 160), -1)
    cv2.ellipse(img, (150, 140), (80, 110), 0, 0, 360, (160, 130, 110), 2)

    # Eyes
    cv2.circle(img, (120, 110), 14, (255, 255, 255), -1)
    cv2.circle(img, (180, 110), 14, (255, 255, 255), -1)
    cv2.circle(img, (120, 110), 6, (80, 50, 20), -1)
    cv2.circle(img, (180, 110), 6, (80, 50, 20), -1)

    # Eyebrows
    cv2.line(img, (105, 90), (135, 92), (60, 40, 20), 3)
    cv2.line(img, (165, 92), (195, 90), (60, 40, 20), 3)

    # Nose
    pts = np.array([[150, 120], [145, 150], [155, 150]], np.int32)
    cv2.polylines(img, [pts], False, (140, 110, 90), 2)

    # Mouth
    cv2.ellipse(img, (150, 180), (30, 15), 0, 0, 180, (50, 50, 180), 3)

    # Hair
    cv2.ellipse(img, (150, 75), (85, 45), 0, 180, 360, (40, 30, 20), -1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Sample test face generated at: {output_path}")

if __name__ == "__main__":
    target_dir = os.path.join(os.path.dirname(__file__), "..", "data", "sample_faces")
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, "target_person.jpg")
    generate_sample_face_image(target_path, "John Doe")
