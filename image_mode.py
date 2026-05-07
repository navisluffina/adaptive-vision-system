import cv2
import numpy as np

from modules.condition import analyze_condition
from modules.enhancement import enhance_frame
from modules.detection import detect

def run_image_mode(image_path):

    # Read image
    image = cv2.imread(image_path)

    if image is None:
        print("❌ Could not load image")
        return

    # Analyze
    condition, brightness, blur = analyze_condition(image)

    # Enhance
    enhanced = enhance_frame(image, condition)

    # Resize both
    image = cv2.resize(image, (640, 480))
    enhanced = cv2.resize(enhanced, (640, 480))

    # Detection
    detected_original = detect(image)

    # Detection on enhanced image
    detected_enhanced = detect(enhanced)

    # Labels
    cv2.putText(
        detected_original,
        "ORIGINAL",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        detected_enhanced,
        "ENHANCED",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # Condition text
    cv2.putText(
        detected_enhanced,
        f"Condition: {condition}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    # Combine
    combined = np.hstack((
        detected_original,
        detected_enhanced
    ))

    # Divider line
    cv2.line(
        combined,
        (640, 0),
        (640, 480),
        (255, 255, 255),
        2
    )

    # Show output
    cv2.imshow(
        "Image Enhancement Mode",
        combined
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()