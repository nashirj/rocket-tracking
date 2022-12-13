"""Helper program to split a video into individual image frames."""

import cv2

capture = cv2.VideoCapture("../data/spacex-rocket-trimmed.mp4")
frameNr = 0
new_shape = (640, 640)

while (True):
    success, frame = capture.read()
    
    if not success:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Saves every 50th frame to use as training data (manually labeled)
    if frameNr % 50 == 0:
        image = cv2.resize(frame, new_shape, interpolation=cv2.INTER_AREA)
        cv2.imwrite(f'../data/all-spacex-images/frame_{frameNr}.jpg', frame)

    frameNr += 1

capture.release()
cv2.destroyAllWindows()
