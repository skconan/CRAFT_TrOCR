import cv2 as cv
import numpy as np
from typing import Tuple, List
from pdf2image import convert_from_path

import mimetypes


def resize_image(image: np.ndarray, size: int = 1024) -> np.ndarray:
    max_size = max(image.shape[:2])
    if max_size <= size:
        return image
    scale = size / max_size
    image = cv.resize(image, (0, 0), fx=scale, fy=scale)
    return image


def check_is_image_or_pdf(file_path: str) -> Tuple[bool, str]:
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type:
        if mime_type.startswith("image"):
            return True, "img"
        elif mime_type == "application/pdf":
            return True, "pdf"
    return False, None


def pdf_to_image_array(pdf_path: str, dpi=200) -> List[np.ndarray]:
    images = convert_from_path(pdf_path, dpi=dpi)
    image_arrays = [resize_image(np.array(image)) for image in images]
    return image_arrays


def read_media(file_path):
    ret, file_type = check_is_image_or_pdf(file_path)

    if not ret:
        raise ValueError("Unsupported file type")
    if file_type == "img":
        img = cv.imread(file_path)
        img = resize_image(img)
        img_list = [img]
    elif file_type == "pdf":
        img_list = pdf_to_image_array(file_path)
    return img_list
