import numpy as np
import cv2


video_path= 'C:/Users/yuhan/Downloads/spiral.avi'
cap = cv2.VideoCapture(video_path)
count = 0
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Display the resulting frame
    cv2.imwrite('frames/%04d.png' % count, frame)
    count += 1

    if cv2.waitKey(1) == ord('q'):
        break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()