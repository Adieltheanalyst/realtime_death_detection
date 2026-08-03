"""
compare_clips.py

Extracts one frame from each clip at a given timestamp, prints their
resolution/FPS metadata, and saves a side-by-side comparison image
so you can visually check whether HUD layouts match before running
the pipeline or the evaluator.

Edit the CLIPS list and SAMPLE_TIME_SEC before running.

Usage:
    python compare_clips.py
"""

import cv2
import numpy as np

# ── Edit these ────────────────────────────────────────────────────────────────
CLIPS = [
    r"data\valorant_test_clip.mp4",   # your reference clip
    r"data\ValorantDeath.mp4",           # the new clip to compare
]
SAMPLE_TIME_SEC = 15   # timestamp to extract from each clip (pick an alive moment)
OUTPUT_IMAGE    = "clip_comparison.png"
DISPLAY_SCALE   = 0.4  # shrink for the comparison image (0.4 = 40% of native size)
# ─────────────────────────────────────────────────────────────────────────────


def extract_frame(clip_path, time_sec):
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {clip_path}")
        return None, {}

    meta = {
        "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps":    cap.get(cv2.CAP_PROP_FPS),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    meta["duration_sec"] = round(meta["frames"] / meta["fps"], 1) if meta["fps"] else 0

    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
    ret, frame = cap.read()
    cap.release()

    return (frame if ret else None), meta


def annotate(frame, clip_path, meta):
    """Burn metadata text onto the frame."""
    out = frame.copy()
    lines = [
        f"{clip_path.split('/')[-1].split(chr(92))[-1]}",
        f"{meta['width']}x{meta['height']} @ {meta['fps']:.0f} FPS",
        f"Duration: {meta['duration_sec']}s",
    ]
    for i, line in enumerate(lines):
        cv2.putText(out, line, (20, 40 + i * 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
        cv2.putText(out, line, (20, 40 + i * 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return out


def main():
    frames = []
    max_h  = 0

    print("Clip comparison\n" + "=" * 50)

    for clip_path in CLIPS:
        print(f"\n{clip_path}")
        frame, meta = extract_frame(clip_path, SAMPLE_TIME_SEC)

        if frame is None:
            print("  Could not extract frame — skipping.")
            continue

        print(f"  Resolution : {meta['width']}x{meta['height']}")
        print(f"  FPS        : {meta['fps']:.0f}")
        print(f"  Duration   : {meta['duration_sec']}s")

        annotated = annotate(frame, clip_path, meta)

        # Resize to display scale
        dw = int(meta["width"]  * DISPLAY_SCALE)
        dh = int(meta["height"] * DISPLAY_SCALE)
        resized = cv2.resize(annotated, (dw, dh))

        frames.append(resized)
        max_h = max(max_h, dh)

    if len(frames) < 2:
        print("\nNeed at least 2 clips to compare. Check file paths.")
        return

    # Pad frames to the same height if resolutions differ
    padded = []
    for f in frames:
        h, w = f.shape[:2]
        if h < max_h:
            pad = np.zeros((max_h - h, w, 3), dtype=np.uint8)
            f = np.vstack([f, pad])
        padded.append(f)

    # Add a thin separator line between clips
    sep = np.full((max_h, 4, 3), 200, dtype=np.uint8)
    comparison = padded[0]
    for p in padded[1:]:
        comparison = np.hstack([comparison, sep, p])

    cv2.imwrite(OUTPUT_IMAGE, comparison)
    print(f"\nSaved {OUTPUT_IMAGE}")
    print("Open it to visually compare HUD layout, panel position, and resolution.")

    # Key things to check visually:
    print("\nWhat to look for:")
    print("  1. Are the resolutions the same? If not, recalibrate before running.")
    print("  2. Does the HUD (health, abilities) sit in the same screen position?")
    print("  3. Is the right side of the screen clear of overlays/facecams?")
    print("  4. Does the map/minimap style look similar? (different maps are fine)")


if __name__ == "__main__":
    main()