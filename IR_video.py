import cv2
import numpy as np

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #resize frame to 1080p
    frame = cv2.resize(frame, (1920, 1080))
    # Display the processed frame:
    cv2.imshow("Processed OBS Window", frame)
    # Break the loop when the 'q' key is pressed:
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the capture and close the window:
cap.release()
cv2.destroyAllWindows()