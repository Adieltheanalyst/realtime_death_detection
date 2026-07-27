import cv2
import shutil
import numpy as np


RAW_FRAME_PATH=r"raw_frame.png"
TEMPLATE_PATH=r"data\death_template.png"
MARGIN_X=60
MARGIN_Y=45

frame=cv2.imread(RAW_FRAME_PATH,0)
template=cv2.imread(TEMPLATE_PATH,0)

if frame is None:
    raise SystemExit(f"Could not read {RAW_FRAME_PATH} - check thi spath.")
if template is None:
    raise SystemExit(f"Could not read {TEMPLATE_PATH} - check the path")

fh,fw= frame.shape
th,tw = template.shape
print(f"Frame: {fw}x{fh} | Current tempate: {tw}x{th}")

best_conf,best_scale,best_loc,best_dims=0.0,1.0,(0,0), (tw,th)
for scale in np.linspace(0.5,1.6,56):
    new_w, new_h=int(tw*scale), int(th*scale)
    if new_w < 20 or new_h < 8 or new_w>fw or new_h > fh:
        continue
    resized= cv2.resize(template,(new_w,new_h))
    result= cv2.matchTemplate(frame,resized,cv2.TM_CCOEFF_NORMED)
    _, max_val,_,max_loc=cv2.minMaxLoc(result)
    if max_val > best_conf:
        best_conf,best_scale = max_val,scale
        best_loc, best_dims=max_loc, (new_w, new_h)


x,y = max_loc
bw,bh= best_dims

print(f"\nBest match: confidence {best_conf:.2f} at scale {best_scale:.2f}")
print(f"\nBanner location in raw_frame: top-left ({x}, {y}), size {bw}x{bh}")

if best_conf < 0.60:
    raise SystemExit(
        "\nWARNING: even the best multi-scale match is weak. Check that "
        "raw_frame.png is the actual death frame (Combat Report visible) "
        "and that the template really shows the banner. Not overwriting "
        "anything."
    )

if abs(best_scale-1.0) > 0.05:
    print(f"\n>>> DIAGNOSIS CONFIRMED: your old template was ~{best_scale:.0%} "
          f"of native size. That is why confidence was stuck around 0.5.")
else:
    print("\nTemplate scale looks fine; refreshing it from raw_frame anyway "
          "for a perfect pixel match.")


new_template = frame[y:y + bh, x:x + bw]
backup=TEMPLATE_PATH.replace(".png","_old.png")
shutil.copy(TEMPLATE_PATH,backup)
cv2.imwrite(TEMPLATE_PATH, new_template)
print(f"\nSaved new native-res template ({bw}x{bh}) to {TEMPLATE_PATH}")
print(f"(old template backed up as {backup})")


x1 = max(0, x - MARGIN_X)
y1 = max(0, y - MARGIN_Y)
x2 = min(fw, x + tw + MARGIN_X)
y2 = min(fh, y + th + MARGIN_Y)
 
print("\nPaste these constants into main.py:")
print(f"ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = {x1}, {y1}, {x2}, {y2}")
 
# Save a visual preview so you can sanity-check the box
preview = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 3)
cv2.rectangle(preview, (x, y), (x + tw, y + th), (0, 0, 255), 2)
cv2.imwrite("roi_preview.png", preview)
print("\nSaved roi_preview.png - green box = ROI, red box = exact banner match.")

result  = cv2.matchTemplate(frame,new_template,cv2.TM_CCOEFF_NORMED)
_, verify_conf, _, _ = cv2.minMaxLoc(result)
print(f"\nVerification: new template matches raw_frame at "
      f"{verify_conf:.2f} confidence (should be ~1.00).")