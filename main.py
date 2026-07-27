import cv2
from collections import deque
import time
import numpy as np



VIDEO_PATH=r"data/valorant_test_clip.mp4"
DEATH_TEMPLATE_PATH = r"data\death_template.png"
TARGET_FPS=4
BUFFER_SECONDS=10
MAX_BUFFER_LENGTH=TARGET_FPS * BUFFER_SECONDS
START_TIME_SEC = 55

ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = 1369, 504, 1875, 635
MATCH_THRESHOLD = 0.75
CONFIRM_FRAMES=2
RESPAWN_CONFIRM_FRAMES=4


DAMAGE_THRESHOLD = 12.0

EDGE_STRIP = 60
DAMAGE_DELTA=4.0
BASELINE_ALPHA=0.05


# HP_X1, HP_Y1, HP_X2, HP_Y2 = 0, 0, 0, 0   # <-- replace
# HP_BRIGHTNESS = 170
event_buffer=deque(maxlen=MAX_BUFFER_LENGTH)

def mock_llm_summarizer(buffer_data):
    """Placeholder for your OpenAI/Gemini API call"""
    print("\n[SYSTEM] Compiling 10-second buffer data...")
    prompt_context= "\n".join(buffer_data)
    time.sleep(1)
    return "Player took damage, dropped to 10 health and was eliminated"

# def estimate_health(gray_frame):
#     """Health %, from how much of the bar's width contains bright pixels."""
#     strip = gray_frame[HP_Y1:HP_Y2, HP_X1:HP_X2]
#     _, binary = cv2.threshold(strip, HP_BRIGHTNESS, 255, cv2.THRESH_BINARY)
#     filled_columns = (binary.max(axis=0) > 0).sum()
#     return int(round(100 * filled_columns / binary.shape[1]))

def damage_redness(frame_bgr):
    """Mean 'redness' of the left/right/bottom screen edges.
    High values mean the red damage vignette is on screen."""
    h, w = frame_bgr.shape[:2]
    s = EDGE_STRIP
    strips = [frame_bgr[:, :s], frame_bgr[:, w - s:], frame_bgr[h - s:, :]]
    score = 0.0
    for st in strips:
        st = st.astype(np.int16)
        redness = np.clip(st[:, :, 2] - (st[:, :, 1] + st[:, :, 0]) // 2, 0, 255)
        score += float(redness.mean())
    return score / len(strips)

def main():
    cap=cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_PATH}")
        return 

    death_template=cv2.imread(DEATH_TEMPLATE_PATH,0)
    if death_template is None :
        print(f"Error: Could not find {DEATH_TEMPLATE_PATH}. Did you save the crop?")
        return 
    # _, template_mask = cv2.threshold(death_template, 200,255, cv2.THRESH_BINARY)

    th,tw = death_template.shape
    if tw> (ROI_X2 - ROI_X1) or th > (ROI_Y2 - ROI_Y1):
        print("Error: ROI is smaller than the template. Re-run calibrate_roi.py")
        return 
    

    # template_edges=cv2.Canny(death_template,50,150)
    # _, binary_template=cv2.threshold(death_template,180,255,cv2.THRESH_BINARY)

    original_fps=cap.get(cv2.CAP_PROP_FPS)
    frame_skip_interval = int(original_fps/TARGET_FPS)
    
    cap.set(cv2.CAP_PROP_POS_MSEC, START_TIME_SEC * 1000)
    frame_count= int(START_TIME_SEC*original_fps)
    
    print(f"Starting pipeline. Original FPS: {original_fps}." 
          f"Processing at {TARGET_FPS} FPS."
          f"({ROI_X1}, {ROI_Y1})-({ROI_X2},{ROI_Y2})")

    cv2.namedWindow("Video Feed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video Feed", 1280, 720)

    # State
    player_state= "ALIVE"
    consecutive_hits=0
    consecutive_misses = 0
    was_taking_damage=False
    redness_baseline = None
    death_count=0

    while True:
        ret,frame=cap.read()
        if not ret:
            print("End of video stream.")
            break

        frame_count += 1

        cv2.imshow("Video Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # DownSampling

        if frame_count % frame_skip_interval != 0:
            continue

        current_time_sec = frame_count / original_fps
        timestamp=f"{current_time_sec:.1f}s"

        # Lightweight CV / METADATA EXTRACTION 
        gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        roi = gray_frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]

        # # _,binary_frame = cv2.threshold(gray_frame,180,255,cv2.THRESH_BINARY)
        # frame_edges= cv2.Canny(gray_frame,50,150)

        # result=cv2.matchTemplate(binary_frame,binary_template,cv2.TM_CCOEFF_NORMED)
        result=cv2.matchTemplate(roi,death_template,cv2.TM_CCOEFF_NORMED)
        _,max_val,_,_=cv2.minMaxLoc(result)

        

        banner_present = max_val >= MATCH_THRESHOLD
        consecutive_hits = consecutive_hits +1 if banner_present else 0
        consecutive_misses=0 if banner_present else consecutive_misses + 1

        redness = damage_redness(frame)
        if redness_baseline is None:
            redness_baseline = redness
        taking_damage = redness >= redness_baseline + DAMAGE_DELTA
        if not taking_damage:
            redness_baseline = ((1- BASELINE_ALPHA)* redness_baseline
                                + BASELINE_ALPHA*redness )
        print(f"[{timestamp}] State: {player_state} | Conf: {max_val:.2f} | "
              f"Redness: {redness:.1f} (base {redness_baseline:.1f})")

        if player_state == "ALIVE":
            if taking_damage and not was_taking_damage:
                event_buffer.append(f"[{timestamp}] EVENT: TOOK DAMAGE")
            else:
                event_buffer.append(f"[{timestamp}] Status: Alive")
            was_taking_damage = taking_damage

            if consecutive_hits >= CONFIRM_FRAMES:
                death_count += 1
                detection_time = time.time()
                event_buffer.append(f"[{timestamp}] Status: DEAD")

                print(f"\n[CV TRIGGER] Death #{death_count} detected at "
                      f"{timestamp} (confidence {max_val:.2f})")
                print("--- BUFFER CONTENTS ---")
                print("\n".join(event_buffer))

                summary= mock_llm_summarizer(list(event_buffer))
                done_time = time.time()
                print(f"\n>>> SUMMARY: {summary}")
                print(f">>> LATENCY: {done_time - detection_time:.2f}s "
                      f"(detection-to-summary)\n")

                event_buffer.clear()
                player_state="DEAD"
                was_taking_damage=False

        elif player_state == "DEAD":
            if consecutive_misses >= RESPAWN_CONFIRM_FRAMES:
                player_state = "ALIVE"
                event_buffer.clear()
                event_buffer.append(f"[{timestamp}] EVENT: RESPAWNED")
                print(f"\n[STATE] Respawn detected at {timestamp}. "
                      f"Buffer flushed - watching for next death.\n")
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nRun complete. Deaths detected and summarized: {death_count}")
    

if __name__ == "__main__":
    main()