import cv2

VIDEO_PATH=r"data/valorant_test_clip.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_MSEC, 62* 1000)

ret,frame = cap.read()
if ret:
    cv2.imwrite("raw_frame.png", frame)
    print("Perfect raw frame saved as 'raw_frame.png'!")
else:
    print("Error reading frame.")

cap.release()