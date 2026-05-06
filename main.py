import cv2
import numpy as np
import os
from datetime import datetime

from modules.condition import analyze_condition
from modules.enhancement import enhance_frame
from modules.detection import detect

# Create output folder
if not os.path.isdir("output/images"):
    os.makedirs("output/images")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    condition, brightness, blur = analyze_condition(frame)
    enhanced = enhance_frame(frame, condition)

    frame_resized = cv2.resize(frame, (640, 480))
    enhanced_resized = cv2.resize(enhanced, (640, 480))

    detected_original = detect(frame_resized)
    detected_enhanced = detect(enhanced_resized)

    # Labels
    cv2.putText(detected_original, "ORIGINAL", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(detected_enhanced, "ENHANCED", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.putText(detected_enhanced, f"Condition: {condition}", (10,70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    cv2.putText(detected_enhanced, f"B:{int(brightness)} Bl:{int(blur)}", (10,100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    status = "ENHANCED" if condition != "NORMAL" else "NO CHANGE"
    cv2.putText(detected_enhanced, f"Status: {status}", (10,130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    combined = np.hstack((detected_original, detected_enhanced))

    # Divider line
    cv2.line(combined, (640, 0), (640, 480), (255,255,255), 2)

    cv2.imshow("Adaptive Vision System", combined)

    # 🔥 SAVE IMAGE when enhancement happens
    if condition != "NORMAL":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"output/images/{timestamp}.jpg", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()