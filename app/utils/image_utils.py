import base64
import io
import cv2
import numpy as np
from PIL import Image


def decode_base64_image(b64_string: str) -> np.ndarray:
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    data = base64.b64decode(b64_string)
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def encode_image_base64(image: np.ndarray, quality: int = 90) -> str:
    _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def resize_image(image: np.ndarray, max_dim: int = 1280) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    return cv2.resize(image, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_LINEAR)


def pad_to_square(image: np.ndarray, size: int = 640) -> np.ndarray:
    h, w = image.shape[:2]
    pad_h = max(0, size - h)
    pad_w = max(0, size - w)
    return cv2.copyMakeBorder(
        image, 0, pad_h, 0, pad_w,
        cv2.BORDER_CONSTANT, value=(114, 114, 114))
