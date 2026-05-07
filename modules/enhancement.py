import cv2
import numpy as np

def enhance_frame(frame, condition):

    enhanced = frame.copy()

    # ===================================
    # LOW LIGHT ENHANCEMENT
    # ===================================

    if "LOW LIGHT" in condition:

        # Mild denoising
        enhanced = cv2.fastNlMeansDenoisingColored(
            enhanced,
            None,
            5,
            5,
            7,
            21
        )

        # LAB color space
        lab = cv2.cvtColor(
            enhanced,
            cv2.COLOR_BGR2LAB
        )

        l, a, b = cv2.split(lab)

        # Softer CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        l = clahe.apply(l)

        lab = cv2.merge((l, a, b))

        enhanced = cv2.cvtColor(
            lab,
            cv2.COLOR_LAB2BGR
        )

        # Gentle brightness adjustment
        enhanced = cv2.convertScaleAbs(
            enhanced,
            alpha=1.15,
            beta=10
        )

    # ===================================
    # BLUR ENHANCEMENT
    # ===================================

    if "BLURRY" in condition:

        # Mild sharpening
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])

        enhanced = cv2.filter2D(
            enhanced,
            -1,
            kernel
        )

        # Edge-preserving smoothing
        enhanced = cv2.bilateralFilter(
            enhanced,
            9,
            75,
            75
        )

        # Slight detail enhancement
        enhanced = cv2.detailEnhance(
            enhanced,
            sigma_s=8,
            sigma_r=0.15
        )

    return enhanced