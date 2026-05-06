import cv2
import numpy as np

def enhance_frame(frame, condition):

    enhanced = frame.copy()

    # 🔹 LOW LIGHT FIX
    if "LOW LIGHT" in condition:
        # Brighten
        enhanced = cv2.convertScaleAbs(enhanced, alpha=2.2, beta=60)
        # Reduce noise
        enhanced = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # 🔹 BLUR FIX
    if "BLURRY" in condition:
        kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)

    return enhanced