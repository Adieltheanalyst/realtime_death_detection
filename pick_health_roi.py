import cv2

VIDEO_PATH = r"data/valorant_test_clip.mp4"
FRAME_TIME_SEC = 55

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_MSEC, FRAME_TIME_SEC*1000)
ret, frame = cap.read()
cap.release()

if not ret:
    raise SystemExit("Could not read the frame - Check VIDEO_PATH / time.")

print("Drag a tight box around the FULL health bar, then press ENTER.")
x,y,w,h = cv2.selectROI("Select health bar (ENTER to confirm)",
                        frame,showCrosshair=True) 

cv2.destroyAllWindows()

if w == 0 or h == 0:
    raise SystemExit("No seection made. Run again and drag a box.")

print("\nPaste these constants into main.py")
print(f"HP-X1, HP_Y1,HP_X2 = {x}, {y}, {x+w}, {y+h}")

crop= frame[y:y +h, x:x + w]
cv2.imwrite("heath_crop.png", crop)

gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
print(f"\nSaved health_crop.png ({w}x{h}).")
print(f"Crop brightness - min: {gray.min()}, max: {gray.max()}, "
      f"mean: {gray.mean():.0f}")
print("If max is well above 170 and mean background is dark, the default "
      "HP_BRIGHTNESS threshold of 170 in main.py should work as-is.")