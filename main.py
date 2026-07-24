import cv2
from collections import deque
import time

VIDEO_PATH="valorant_test_clip.mp4"
TARGET_FPS=4
BUFFER_SECONDS=10
MAX_BUFFER_LENGTH=TARGET_FPS * BUFFER_SECONDS


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

    original_fps=cap.get(cv2.CAP_PROP_FPS)
    frame_skip_interval = int(original_fps/TARGET_FPS)

    frame_count=0
    print(f"Starting pipeline. Original FPS: {original_fps}. Processing at {TARGET_FPS} FPS.")

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

        mock_health=100
        mock_status = "Alive"

        metadata_entry=f"[{timestamp}] Health: {mock_health} | Status: {mock_status}"
        event_buffer.append(metadata_entry)

        # Trigger Detection 
        if current_time_sec >= 15.0:
            print(f"\n[TRIGGER] Death detected at {timestamp}!")

            start_time=time.time()
            summary=mock_llm_summarizer(list(event_buffer))
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