import cv2
import numpy as np
import os
import time
from datetime import datetime

from image_mode import run_image_mode
from modules.condition import analyze_condition
from modules.enhancement import enhance_frame
from modules.detection import detect

# Create output folder
if not os.path.isdir("output/images"):
    os.makedirs("output/images")

# =========================
# SELECT MODE
# =========================

print("\nAdaptive Vision System")
print("1. Camera Mode")
print("2. Image Mode")

choice = input("Enter choice (1 or 2): ")

# =========================
# IMAGE MODE
# =========================

if choice == "2":

    path = input("Enter image path: ")

    if not os.path.exists(path):
        print("❌ File not found")
    else:
        run_image_mode(path)

# =========================
# CAMERA MODE
# =========================

elif choice == "1":

    cap = cv2.VideoCapture(0)

    last_saved = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            continue

        # Analyze
        condition, brightness, blur = analyze_condition(frame)

        # Enhance
        enhanced = enhance_frame(frame, condition)

        # Resize
        frame_resized = cv2.resize(frame, (640, 480))
        enhanced_resized = cv2.resize(enhanced, (640, 480))

        # Detection
        detected_original = detect(frame_resized)

        # Copy detection
        detected_enhanced = detected_original.copy()

        # Labels
        cv2.putText(detected_original, "ORIGINAL", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(detected_enhanced, "ENHANCED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.putText(detected_enhanced,
                    f"Condition: {condition}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2)

        cv2.putText(detected_enhanced,
                    f"B:{int(brightness)} Bl:{int(blur)}",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2)

        status = "ENHANCED" if condition != "NORMAL" else "NO CHANGE"

        cv2.putText(detected_enhanced,
                    f"Status: {status}",
                    (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2)

        # Combine
        combined = np.hstack((detected_original,
                              detected_enhanced))

        # Divider
        cv2.line(combined,
                 (640, 0),
                 (640, 480),
                 (255, 255, 255),
                 2)

        # Show
        cv2.imshow("Adaptive Vision System", combined)

        # Save every 3 sec
        if condition != "NORMAL":

            if time.time() - last_saved > 3:

                timestamp = datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )

                cv2.imwrite(
                    f"output/images/{timestamp}.jpg",
                    combined
                )

                last_saved = time.time()

        # Quit
        key = cv2.waitKey(10)

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# =========================
# INVALID OPTION
# =========================

else:
    print("❌ Invalid choice")