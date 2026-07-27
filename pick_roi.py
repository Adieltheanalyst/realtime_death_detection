"""
Extracts a frame , saves the crop as a template image
prints the ROI constants (crop +margin) for main.py

"""

import cv2

VIDEO_PATH = r"data/valorant_test_clip.mp4"
FRAME_TIME_SEC = 55                      # a moment where the player is ALIVE
OUT_TEMPLATE = r"data\hud_template.png"  # where to save the crop
MARGIN = 30                              # ROI margin around the crop (px)
 
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_MSEC, FRAME_TIME_SEC * 1000)
ret, frame = cap.read()
cap.release()
 
if not ret:
    raise SystemExit("Could not read the frame - check VIDEO_PATH / time.")
 
fh, fw = frame.shape[:2]
print("Drag a tight box around the element, then press ENTER.")
x, y, w, h = cv2.selectROI("Select region (ENTER to confirm)",
                           frame, showCrosshair=True)
cv2.destroyAllWindows()
 
if w == 0 or h == 0:
    raise SystemExit("No selection made. Run again and drag a box.")
 
crop = frame[y:y + h, x:x + w]
cv2.imwrite(OUT_TEMPLATE, crop)
print(f"\nSaved template ({w}x{h}) to {OUT_TEMPLATE}")
 
x1 = max(0, x - MARGIN)
y1 = max(0, y - MARGIN)
x2 = min(fw, x + w + MARGIN)
y2 = min(fh, y + h + MARGIN)
print("\nPaste these constants into main.py:")
print(f"HUD_X1, HUD_Y1, HUD_X2, HUD_Y2 = {x1}, {y1}, {x2}, {y2}")