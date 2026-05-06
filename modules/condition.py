import cv2
import numpy as np
from config import BRIGHTNESS_THRESHOLD, BLUR_THRESHOLD

def analyze_condition(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    brightness = np.mean(gray)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()

    condition = "NORMAL"

    if brightness < BRIGHTNESS_THRESHOLD:
        condition = "LOW LIGHT"
    elif blur < BLUR_THRESHOLD:
        condition = "BLURRY"

    return condition, brightness, blur