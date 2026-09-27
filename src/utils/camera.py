import os

import cv2


def open_default_camera(preferred_indices=(0, 1, 2, 3)):
    camera_index_env = os.getenv("SIGNDETR_CAMERA_INDEX")
    if camera_index_env is not None:
        try:
            preferred_indices = (int(camera_index_env),)
        except ValueError:
            print(f"[WARN] Invalid SIGNDETR_CAMERA_INDEX={camera_index_env!r}; falling back to defaults")

    backends = []
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append(cv2.CAP_DSHOW)
    backends.append(None)

    for index in preferred_indices:
        for backend in backends:
            cap = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
            if cap.isOpened():
                print(f"[CAMERA] Using camera index {index}")
                return cap
            cap.release()

    print(f"[ERROR] Could not open camera. Tried indexes: {preferred_indices}")
    return cv2.VideoCapture()