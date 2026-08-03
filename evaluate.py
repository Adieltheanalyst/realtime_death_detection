
import cv2
import json
import time
import numpy as np
from collections import deque
from pathlib import Path
 
# ── Paste your calibrated constants here ──────────────────────────────────────
ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = 1300, 250, 1920, 800
HUD_X1, HUD_Y1, HUD_X2, HUD_Y2 = 698, 1026, 1248, 1080
DEATH_TEMPLATE_PATH = r"data\death_template.png"
HUD_TEMPLATE_PATH   = r"data\hud_template.png"
# ──────────────────────────────────────────────────────────────────────────────
 
TARGET_FPS             = 4
BUFFER_SECONDS         = 10
MAX_BUFFER_LENGTH      = TARGET_FPS * BUFFER_SECONDS
MATCH_THRESHOLD        = 0.75
BANNER_CLEAR_THRESHOLD = 0.50
HUD_MATCH_THRESHOLD    = 0.70
CONFIRM_FRAMES         = 3
RESPAWN_CONFIRM_FRAMES = 6
EDGE_STRIP             = 60
DAMAGE_DELTA           = 4.0
BASELINE_ALPHA         = 0.05
 
# How many seconds of tolerance when matching detected vs ground-truth events.
# A detection within this window counts as a True Positive.
TOLERANCE_SEC = 2.0
 
GROUND_TRUTH_FILE = "ground_truth.json"
REPORT_FILE       = "evaluation_report.json"
 
 
# ── CV helpers (same as main.py) ──────────────────────────────────────────────
 
def roi_match(gray_frame, template, x1, y1, x2, y2):
    roi = gray_frame[y1:y2, x1:x2]
    result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return max_val, max_loc, roi
 
 
def validated_death_conf(gray_frame, death_template):
    conf, loc, roi = roi_match(gray_frame, death_template,
                               ROI_X1, ROI_Y1, ROI_X2, ROI_Y2)
    th, tw = death_template.shape
    x, y = loc
    patch = roi[y:y + th, x:x + tw]
    if patch.size == 0 or patch.std() < 15 or patch.max() < 150:
        return 0.0
    return conf
 
 
def damage_redness(frame_bgr):
    h, w = frame_bgr.shape[:2]
    s = EDGE_STRIP
    strips = [frame_bgr[:, :s], frame_bgr[:, w - s:], frame_bgr[h - s:, :]]
    score = 0.0
    for st in strips:
        st = st.astype(np.int16)
        redness = np.clip(
            st[:, :, 2] - (st[:, :, 1] + st[:, :, 0]) // 2, 0, 255)
        score += float(redness.mean())
    return score / len(strips)
 
 
# ── Matching logic ────────────────────────────────────────────────────────────
 
def match_events(detected, ground_truth, tolerance=TOLERANCE_SEC):
    """
    Greedy nearest-match between detected timestamps and ground truth.
    Returns (true_positives, false_positives, false_negatives).
    """
    matched_gt = set()
    matched_det = set()
 
    for i, det in enumerate(detected):
        for j, gt in enumerate(ground_truth):
            if j in matched_gt:
                continue
            if abs(det - gt) <= tolerance:
                matched_gt.add(j)
                matched_det.add(i)
                break
 
    tp = len(matched_gt)
    fp = len(detected) - len(matched_det)
    fn = len(ground_truth) - tp
    return tp, fp, fn
 
 
def safe_div(a, b):
    return a / b if b > 0 else 0.0
 
 
# ── Per-clip pipeline ─────────────────────────────────────────────────────────
 
def run_clip(clip_path, death_template, hud_template):
    """Run detection on one clip. Returns detected death/respawn timestamps."""
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {clip_path}")
        return [], []
 
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip   = int(original_fps / TARGET_FPS)
    frame_count  = 0
 
    player_state       = "ALIVE"
    consecutive_deaths = 0
    consecutive_hud    = 0
    was_taking_damage  = False
    redness_baseline   = None
    event_buffer       = deque(maxlen=MAX_BUFFER_LENGTH)
 
    detected_deaths   = []
    detected_respawns = []
 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue
 
        current_time = frame_count / original_fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
        death_conf = validated_death_conf(gray, death_template)
        hud_conf, _, _ = roi_match(gray, hud_template,
                                   HUD_X1, HUD_Y1, HUD_X2, HUD_Y2)
 
        banner_present = death_conf >= MATCH_THRESHOLD
        banner_maybe   = death_conf >= BANNER_CLEAR_THRESHOLD
        hud_present    = hud_conf   >= HUD_MATCH_THRESHOLD
 
        consecutive_deaths = consecutive_deaths + 1 if banner_present else 0
        consecutive_hud    = (consecutive_hud + 1
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
 
        timestamp = f"{current_time:.1f}s"
        if player_state == "ALIVE":
            if taking_damage and not was_taking_damage:
                event_buffer.append(f"[{timestamp}] EVENT: TOOK DAMAGE")
            else:
                event_buffer.append(f"[{timestamp}] Status: Alive")
            was_taking_damage = taking_damage
 
            if consecutive_deaths >= CONFIRM_FRAMES:
                detected_deaths.append(round(current_time, 1))
                event_buffer.clear()
                player_state       = "DEAD"
                was_taking_damage  = False
                consecutive_hud    = 0
 
        elif player_state == "DEAD":
            if consecutive_hud >= RESPAWN_CONFIRM_FRAMES:
                detected_respawns.append(round(current_time, 1))
                player_state = "ALIVE"
                event_buffer.clear()
                event_buffer.append(f"[{timestamp}] EVENT: RESPAWNED")
 
    cap.release()
    return detected_deaths, detected_respawns
 
 
# ── Main evaluation loop ──────────────────────────────────────────────────────
 
def main():
    # Load templates
    death_template = cv2.imread(DEATH_TEMPLATE_PATH, 0)
    hud_template   = cv2.imread(HUD_TEMPLATE_PATH, 0)
    if death_template is None:
        raise SystemExit(f"Cannot load {DEATH_TEMPLATE_PATH}")
    if hud_template is None:
        raise SystemExit(f"Cannot load {HUD_TEMPLATE_PATH}")
 
    # Load ground truth
    gt_path = Path(GROUND_TRUTH_FILE)
    if not gt_path.exists():
        raise SystemExit(
            f"{GROUND_TRUTH_FILE} not found.\n"
            "Create it manually with death/respawn timestamps per clip.\n"
            "Format: {\"clip1.mp4\": {\"deaths\": [62.0, 189.5], \"respawns\": [98.0]}}"
        )
    with open(gt_path) as f:
        ground_truth = json.load(f)
 
    report = {}
    total_death_tp = total_death_fp = total_death_fn = 0
    total_resp_tp  = total_resp_fp  = total_resp_fn  = 0
 
    for clip_name, gt in ground_truth.items():
        clip_path = str(Path("data") / clip_name)
        print(f"\nProcessing: {clip_name}")
        t0 = time.time()
 
        det_deaths, det_respawns = run_clip(
            clip_path, death_template, hud_template)
 
        elapsed = time.time() - t0
        gt_deaths   = gt.get("deaths", [])
        gt_respawns = gt.get("respawns", [])
 
        d_tp, d_fp, d_fn = match_events(det_deaths,   gt_deaths)
        r_tp, r_fp, r_fn = match_events(det_respawns, gt_respawns)
 
        d_precision = safe_div(d_tp, d_tp + d_fp)
        d_recall    = safe_div(d_tp, d_tp + d_fn)
        r_precision = safe_div(r_tp, r_tp + r_fp)
        r_recall    = safe_div(r_tp, r_tp + r_fn)
 
        total_death_tp += d_tp;  total_death_fp += d_fp;  total_death_fn += d_fn
        total_resp_tp  += r_tp;  total_resp_fp  += r_fp;  total_resp_fn  += r_fn
 
        clip_report = {
            "processing_time_sec": round(elapsed, 1),
            "deaths": {
                "ground_truth":     gt_deaths,
                "detected":         det_deaths,
                "true_positives":   d_tp,
                "false_positives":  d_fp,
                "false_negatives":  d_fn,
                "precision":        round(d_precision, 3),
                "recall":           round(d_recall, 3),
            },
            "respawns": {
                "ground_truth":     gt_respawns,
                "detected":         det_respawns,
                "true_positives":   r_tp,
                "false_positives":  r_fp,
                "false_negatives":  r_fn,
                "precision":        round(r_precision, 3),
                "recall":           round(r_recall, 3),
            },
        }
        report[clip_name] = clip_report
 
        print(f"  Deaths   — GT: {gt_deaths} | Detected: {det_deaths}")
        print(f"             TP={d_tp}  FP={d_fp}  FN={d_fn} | "
              f"Precision={d_precision:.0%}  Recall={d_recall:.0%}")
        print(f"  Respawns — GT: {gt_respawns} | Detected: {det_respawns}")
        print(f"             TP={r_tp}  FP={r_fp}  FN={r_fn} | "
              f"Precision={r_precision:.0%}  Recall={r_recall:.0%}")
        print(f"  Clip processed in {elapsed:.0f}s")
 
    # Aggregate
    agg_d_prec = safe_div(total_death_tp, total_death_tp + total_death_fp)
    agg_d_rec  = safe_div(total_death_tp, total_death_tp + total_death_fn)
    agg_r_prec = safe_div(total_resp_tp,  total_resp_tp  + total_resp_fp)
    agg_r_rec  = safe_div(total_resp_tp,  total_resp_tp  + total_resp_fn)
 
    aggregate = {
        "clips_evaluated": len(ground_truth),
        "tolerance_sec":   TOLERANCE_SEC,
        "deaths": {
            "total_ground_truth":  total_death_tp + total_death_fn,
            "total_detected":      total_death_tp + total_death_fp,
            "true_positives":      total_death_tp,
            "false_positives":     total_death_fp,
            "false_negatives":     total_death_fn,
            "aggregate_precision": round(agg_d_prec, 3),
            "aggregate_recall":    round(agg_d_rec, 3),
        },
        "respawns": {
            "total_ground_truth":  total_resp_tp + total_resp_fn,
            "total_detected":      total_resp_tp + total_resp_fp,
            "true_positives":      total_resp_tp,
            "false_positives":     total_resp_fp,
            "false_negatives":     total_resp_fn,
            "aggregate_precision": round(agg_r_prec, 3),
            "aggregate_recall":    round(agg_r_rec, 3),
        },
    }
    report["_aggregate"] = aggregate
 
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)
 
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)
    print(f"Clips evaluated : {aggregate['clips_evaluated']}")
    print(f"Tolerance window: ±{TOLERANCE_SEC}s")
    print()
    print(f"DEATH DETECTION")
    print(f"  Ground truth   : {aggregate['deaths']['total_ground_truth']}")
    print(f"  Detected       : {aggregate['deaths']['total_detected']}")
    print(f"  True positives : {aggregate['deaths']['true_positives']}")
    print(f"  False positives: {aggregate['deaths']['false_positives']}")
    print(f"  False negatives: {aggregate['deaths']['false_negatives']}")
    print(f"  Precision      : {agg_d_prec:.0%}")
    print(f"  Recall         : {agg_d_rec:.0%}")
    print()
    print(f"RESPAWN DETECTION")
    print(f"  Ground truth   : {aggregate['respawns']['total_ground_truth']}")
    print(f"  Detected       : {aggregate['respawns']['total_detected']}")
    print(f"  True positives : {aggregate['respawns']['true_positives']}")
    print(f"  False positives: {aggregate['respawns']['false_positives']}")
    print(f"  False negatives: {aggregate['respawns']['false_negatives']}")
    print(f"  Precision      : {agg_r_prec:.0%}")
    print(f"  Recall         : {agg_r_rec:.0%}")
    print()
    print(f"Full report saved to {REPORT_FILE}")
 
 
if __name__ == "__main__":
    main()
 