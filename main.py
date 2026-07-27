import cv2
from collections import deque
import time
import math 


VIDEO_PATH=r"data/valorant_test_clip.mp4"
DEATH_TEMPLATE_PATH = r"data\death_template.png"
TARGET_FPS=4
BUFFER_SECONDS=10
MAX_BUFFER_LENGTH=TARGET_FPS * BUFFER_SECONDS
MATCH_THRESHOLD = 0.75

CONFIRM_FRAMES=2

ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = 1369, 504, 17296, 627


event_buffer=deque(maxlen=MAX_BUFFER_LENGTH)

def mock_llm_summarizer(buffer_data):
    """Placeholder for your OpenAI/Gemini API call"""
    print("\n[SYSTEM] Compiling 10-second buffer data...")
    prompt_context= "\n".join(buffer_data)

    time.sleep(1)
    return "Player took damager, dropped to 10 health and was eliminated"

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
    
    start_time_sec = 55
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time_sec * 1000)
    frame_count= int(start_time_sec*original_fps)
    
    print(f"Starting pipeline. Original FPS: {original_fps}." 
          f"Processing at {TARGET_FPS} FPS."
          f"({ROI_X1}, {ROI_Y1})-({ROI_X2},{ROI_Y2})")

    cv2.namedWindow("Video Feed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video Feed", 1280, 720)

    consecutive_hits=0

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

        current_time_sec = frame_count/ original_fps
        timestamp=f"{current_time_sec:.1f}s"

        # Lightweight CV / METADATA EXTRACTION 
        gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        roi = gray_frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]

        # # _,binary_frame = cv2.threshold(gray_frame,180,255,cv2.THRESH_BINARY)
        # frame_edges= cv2.Canny(gray_frame,50,150)

        # result=cv2.matchTemplate(binary_frame,binary_template,cv2.TM_CCOEFF_NORMED)
        result=cv2.matchTemplate(roi,death_template,cv2.TM_CCOEFF_NORMED)
        _,max_val,_,_=cv2.minMaxLoc(result)

        

        # if math.isinf(max_val) or math.isnan(max_val):
        #     max_val=0.0
        print(f"[{timestamp}] ROI Confidence: {max_val:.2f}")


        if max_val >= MATCH_THRESHOLD:
            consecutive_hits += 1
        else:
            consecutive_hits = 0

        status = "DEAD" if consecutive_hits>= CONFIRM_FRAMES else "Alive"

        mock_health=100
        event_buffer.append(f"[{timestamp}] Health: {mock_health} | Status: {status}")

        if status == "DEAD":
            detection_time = time.time()
            print(f"\n[CV TRIGGER] Death detected at {timestamp}"
                  f"(confidence {max_val:.2f}, confirmed over" 
                  f"{CONFIRM_FRAMES} frames)")
            summary= mock_llm_summarizer(list(event_buffer))
            done_time = time.time()

            print(f"\n>>> FINAL OUTPUT: {summary}")
            print(f">>> SUMMARIZATION LATENCY: {done_time - detection_time:.2f}s "
                  f"(detection-to-summary)")
            break

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()