import cv2
from collections import deque
import time
import numpy as np
import os
from google import genai
from dotenv import load_dotenv
load_dotenv()




VIDEO_PATH=r"data/valorant_test_clip_3.mp4"

DEATH_TEMPLATE_PATH = r"data\death_template.png"
HUD_TEMPLATE_PATH=r"data\hud_template.png"
TARGET_FPS=4
BUFFER_SECONDS=10
MAX_BUFFER_LENGTH=TARGET_FPS * BUFFER_SECONDS
START_TIME_SEC = 50

ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = 1300, 250, 1920, 800
MATCH_THRESHOLD = 0.75
CONFIRM_FRAMES=3

HUD_X1, HUD_Y1, HUD_X2, HUD_Y2 = 698, 1026, 1248, 1080

HUD_MATCH_THRESHOLD=0.70

RESPAWN_CONFIRM_FRAMES=6


DAMAGE_THRESHOLD = 12.0

EDGE_STRIP = 60
DAMAGE_DELTA=4.0
BASELINE_ALPHA=0.05
BANNER_CLEAR_THRESHOLD = 0.50

DEBUG_SNAPSHOTS = True
SNAPSHOT_DIR = "debug_snapshots"


LLM_MODEL = "gemini-3.6-flash"

SUMMARIZER_PROMPT = """You are an esports analyst. Below is a metadata log \
from the final seconds before a player died in Valorant. Each line has a \
video timestamp. TOOK DAMAGE means the player was hit. RESPAWNED at the \
start means the player had just respawned (a very short log means they \
died almost immediately after spawning - say so). No damage events before \
death means they were likely killed instantly.

Write ONE or TWO short sentences explaining how the death played out, \
using the timing (e.g. how long between last damage and death). No \
preamble, no bullet points.

EVENT LOG:
{log}"""

def fallback_summary(buffer_data):
    """Rule based summary if the API is unavailable - keeps the demo alive"""
    damage_times=[ln.split("]")[0].strip("[") for ln in buffer_data
                  if "TOOK DAMAGE" in ln]

    if not damage_times:
        return "Player was eliminated with no prior damage recorded - likely an instant kill."
    return (f"Player took damage {len(damage_times)} time(s)"
            f"(last at {damage_times[-1]}) before being eliminated.")


# HP_X1, HP_Y1, HP_X2, HP_Y2 = 0, 0, 0, 0   # <-- replace
# HP_BRIGHTNESS = 170
event_buffer=deque(maxlen=MAX_BUFFER_LENGTH)

def llm_summarizer(buffer_data):
    prompt = SUMMARIZER_PROMPT.format(log="\n".join(buffer_data))
    try:
        client = genai.Client()
        response= client.models.generate_content(
            model=LLM_MODEL, contents=prompt
        )
        return response.text.strip()
    except Exception as e: 
        print(f"[Warn] LLM call failed ({e}); using rule-based fallback.")
        return fallback_summary(buffer_data)


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

def roi_match(gray_frame,template, x1,y1,x2,y2):
    """Best template- match confidence inside a rectangular ROI."""
    roi = gray_frame[y1:y2, x1:x2]
    result = cv2.matchTemplate(roi, template,cv2.TM_CCOEFF_NORMED)
    _,max_val,_,max_loc = cv2.minMaxLoc(result)
    return max_val,max_loc,roi

def validated_death_conf(gray_frame, template):

    conf,loc, roi = roi_match(gray_frame, template,
                              ROI_X1, ROI_Y1, ROI_X2, ROI_Y2)

    th, tw = template.shape
    x,y = loc
    patch=roi[y:y + th, x:x + tw]
    if patch.size == 0 or patch.std() < 15 or patch.max() < 150:
        return 0.0
    return conf

def save_snapshot(frame, label, timestamp, death_conf, hud_conf):
    if not DEBUG_SNAPSHOTS:
        return
    annotated = frame.copy()
    cv2.putText(annotated,
                f"{label} @ {timestamp}  Death:{death_conf:.2f}  HUD:{hud_conf:.2f}",
                (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.rectangle(annotated, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 0, 255), 2)
    cv2.rectangle(annotated, (HUD_X1, HUD_Y1), (HUD_X2, HUD_Y2), (0, 255, 0), 2)
    fname = f"{SNAPSHOT_DIR}/{label}_{timestamp.replace('.', '_')}.png"
    cv2.imwrite(fname, annotated)


def main():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    cap=cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_PATH}")
        return 

    death_template=cv2.imread(DEATH_TEMPLATE_PATH,0)
    if death_template is None :
        print(f"Error: Could not find {DEATH_TEMPLATE_PATH}. Did you save the crop?")
        return 

    hud_template= cv2.imread(HUD_TEMPLATE_PATH,0)
    if hud_template is None:
        print(f"Error: Could not find {HUD_TEMPLATE_PATH}."
              f"Run pick_roi.py to create it.")
        return 
    if HUD_X2 <= HUD_X1 or HUD_Y2 <= HUD_Y1:
        print("Error: HUD ROI not set. Run pick_roi.py and paste the "
              "printed HUD constants into main.py.")
        return


    

    
    original_fps=cap.get(cv2.CAP_PROP_FPS)
    frame_skip_interval = int(original_fps/TARGET_FPS)
    
    cap.set(cv2.CAP_PROP_POS_MSEC, START_TIME_SEC * 1000)
    frame_count= int(START_TIME_SEC*original_fps)
    
    print(f"Starting pipeline. Original FPS: {original_fps:.0f} FPS, " 
          f"Processing at {TARGET_FPS} FPS.")

    cv2.namedWindow("Video Feed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video Feed", 1280, 720)

    # State
    player_state= "ALIVE"
    consecutive_deaths=0
    consecutive_hud=0
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
        # roi = gray_frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]

        death_conf = validated_death_conf(gray_frame, death_template)

        hud_conf,_,_ = roi_match(gray_frame, hud_template,
                             HUD_X1, HUD_Y1, HUD_X2, HUD_Y2)
        



        # # _,binary_frame = cv2.threshold(gray_frame,180,255,cv2.THRESH_BINARY)
        # frame_edges= cv2.Canny(gray_frame,50,150)

        # result=cv2.matchTemplate(binary_frame,binary_template,cv2.TM_CCOEFF_NORMED)
        # result=cv2.matchTemplate(roi,death_template,cv2.TM_CCOEFF_NORMED)
        # _,max_val,_,_=cv2.minMaxLoc(result)

        

        banner_present = death_conf >= MATCH_THRESHOLD
        banner_maybe= death_conf >= BANNER_CLEAR_THRESHOLD
        hud_present = hud_conf >= HUD_MATCH_THRESHOLD
        consecutive_deaths=consecutive_deaths + 1 if banner_present else 0
        consecutive_hud = (consecutive_hud + 1 
                           if (hud_present and not banner_maybe) else 0)

        redness = damage_redness(frame)
        if redness_baseline is None:
            redness_baseline = redness
        taking_damage = (hud_present and
                         redness >= redness_baseline + DAMAGE_DELTA)

        if hud_present:
            alpha = BASELINE_ALPHA if not taking_damage else 0.01
            redness_baseline = ((1 - alpha) * redness_baseline
                                + alpha * redness)
        print(f"[{timestamp}] {player_state} | Death: {death_conf:.2f} | "
              f"HUD: {hud_conf:.2f} | Red: {redness:.1f}" 
              f"(base {redness_baseline:.1f})")
        

        
        if player_state == "ALIVE":
            if taking_damage and not was_taking_damage:
                event_buffer.append(f"[{timestamp}] EVENT: TOOK DAMAGE")
            else:
                event_buffer.append(f"[{timestamp}] Status: Alive")
            was_taking_damage = taking_damage

            if consecutive_deaths >= CONFIRM_FRAMES:
                death_count += 1
                detection_time = time.time()
                event_buffer.append(f"[{timestamp}] Status: DEAD")

                print(f"\n[CV TRIGGER] Death #{death_count} detected at "
                      f"{timestamp} (confidence {death_conf:.2f})")
                save_snapshot(frame, f"death{death_count}", timestamp,
                              death_conf, hud_conf)
                print("--- BUFFER CONTENTS ---")
                print("\n".join(event_buffer))

                summary= llm_summarizer(list(event_buffer))
                done_time = time.time()
                print(f"\n>>> SUMMARY: {summary}")
                print(f">>> LATENCY: {done_time - detection_time:.2f}s "
                      f"(detection-to-summary)\n")

                event_buffer.clear()
                player_state="DEAD"
                was_taking_damage=False
                consecutive_hud=0

        elif player_state == "DEAD":
            if consecutive_hud >= RESPAWN_CONFIRM_FRAMES:
                player_state = "ALIVE"
                event_buffer.clear()
                event_buffer.append(f"[{timestamp}] EVENT: RESPAWNED")
                save_snapshot(frame, "respawn", timestamp,
                              death_conf, hud_conf)
                print(f"\n[STATE] Respawn detected at {timestamp}. "
                      f"(HUD back). Buffer flushed.\n")
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nRun complete. Deaths detected and summarized: {death_count}")
    

if __name__ == "__main__":
    main()