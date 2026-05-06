import cv2

def show_info(frame, condition):
    cv2.putText(frame, f"Condition: {condition}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)
    return frame