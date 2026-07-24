import cv2
from collections import deque
import time

VIDEO_PATH=r"data/valorant_test_clip.mp4"
DEATH_TEMPLATE_PATH = r"data\death_template.png"
TARGET_FPS=4
BUFFER_SECONDS=10
MAX_BUFFER_LENGTH=TARGET_FPS * BUFFER_SECONDS
MACTH_THRESHOLD = 0.41
import math 


event_buffer=deque(maxlen=MAX_BUFFER_LENGTH)

def mock_llm_summarizer(buffer_data):
    """Placeholder for your OpenAI/Gemini API call"""
    print("\n[SYSTEM] Compiling 10-second buffer data...")
    prompt_cotext= "\n".join(buffer_data)

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

    template_edges=cv2.Canny(death_template,50,150)
    # _, binary_template=cv2.threshold(death_template,180,255,cv2.THRESH_BINARY)

    original_fps=cap.get(cv2.CAP_PROP_FPS)
    frame_skip_interval = int(original_fps/TARGET_FPS)
    
    start_time_sec = 55
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time_sec * 1000)
    frame_count= int(start_time_sec*original_fps)
    
    print(f"Starting pipeline. Original FPS: {original_fps}. Processing at {TARGET_FPS} FPS.")

    cv2.namedWindow("Video Feed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video Feed", 1280, 720)

    while True:
        ret,frame=cap.read()
        if not ret:
            print("End of video stream.")
            break

        frame_count += 1

        # DownSampling

        if frame_count % frame_skip_interval != 0:
            continue
        current_time_sec = frame_count/ original_fps
        timestamp=f"{current_time_sec:.1f}s"

        # Lightweight CV / METADATA EXTRACTION 
        gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        # _,binary_frame = cv2.threshold(gray_frame,180,255,cv2.THRESH_BINARY)
        frame_edges= cv2.Canny(gray_frame,50,150)

        # result=cv2.matchTemplate(binary_frame,binary_template,cv2.TM_CCOEFF_NORMED)
        result=cv2.matchTemplate(frame_edges,template_edges,cv2.TM_CCOEFF_NORMED)
        min_val,max_val,min_loc,max_loc=cv2.minMaxLoc(result)

        if math.isinf(max_val) or math.isnan(max_val):
            max_val=0.0
        print(f"[{timestamp}] Checking template... Confidence: {max_val:.2f}")


        if max_val >= MACTH_THRESHOLD:
            mock_status = "DEAD"
        else:
            mock_status="Alive"
        mock_health=100



        metadata_entry=f"[{timestamp}] Health: {mock_health} | Status: {mock_status}"
        event_buffer.append(metadata_entry)

        cv2.imshow("Video Feed", frame)

        if mock_status == "DEAD":
            print(f"\n[CV TRIGGER] Death icon detected at {timestamp} with {max_val:.2f} confidence!")
            start_time=time.time()
            summary= mock_llm_summarizer(list(event_buffer))
            end_time = time.time()

            print(f"\n>>> FINAL OUTPUT: {summary}")
            print(f">>> LATENCY: {(end_time- start_time):.2f} seconds")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()