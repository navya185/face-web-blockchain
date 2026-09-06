"""
Custom exceptions for face detection and identification.
"""

class FaceProcessingError(Exception):
    """Base exception for face processing errors."""
    pass

class NoFaceFoundError(FaceProcessingError):
    """Raised when no face is detected in an image."""
    def __init__(self, message: str = "No face was detected in the provided image."):
        super().__init__(message)

class MultipleFacesFoundError(FaceProcessingError):
    """Raised when multiple faces are detected when exactly one was expected."""
    def __init__(self, face_count: int, message: str = None):
        if not message:
            message = f"Multiple faces detected ({face_count}). Exactly one face is required."
        self.face_count = face_count
        super().__init__(message)
