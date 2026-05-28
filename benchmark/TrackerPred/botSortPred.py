import sys
from turtle import width
import time
from pathlib import Path
from benchmark.utils.mot_out import write_mot_output, byte_style_to_out
from benchmark.utils.seqinfo import read_seqinfo, find_sequences
from benchmark.utils.load_det import load_detections, get_frame_dets
import numpy as np
from types import SimpleNamespace

#NOTE: use botsort env
# Path setup
BEP_ROOT = Path(__file__).resolve().parents[2]

BOTSORT_REPO = BEP_ROOT / "BoT-SORT"
sys.path.insert(0, str(BOTSORT_REPO))
from tracker.bot_sort import BoTSORT

#Changeable
TRACKER_NAME = "botsort"
SPLIT = "MFT-test"

METRICS = BEP_ROOT / "trackeval"
GT_ROOT = METRICS / "gt" / SPLIT
OUT_DIR = METRICS / "trackers" / SPLIT / TRACKER_NAME / "data"
DET_ROOT = BEP_ROOT / "benchmark" / "dets" / SPLIT

def run_tracking(seq_dir):
    args = SimpleNamespace(
    track_high_thresh=0.4,
    track_low_thresh=0.1,
    new_track_thresh=0.4,

    track_buffer=40,
    match_thresh=0.9,

    proximity_thresh=0.5,
    appearance_thresh=0.25,

    with_reid=False,
    fast_reid_config="",
    fast_reid_weights="",

    cmc_method="none",

    name="MFT",
    ablation=False,
    mot20=False,
    device="cpu"
    )
        
    tracker = BoTSORT(args=args, frame_rate=25)
    width, height, seqlen = read_seqinfo(seq_dir)

    det_file = DET_ROOT / f"{seq_dir.name}.txt"
    det_by_frame = load_detections(det_file)
    
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    pred_file = OUT_DIR / f"{seq_dir.name}.txt"
    dummy_img = np.zeros((height, width, 3), dtype=np.uint8)
    #Initialize time
    seq_time = 0

    with pred_file.open("w", encoding="utf-8") as f:
        for frame_id in range(1, seqlen + 1):
            all_dets = get_frame_dets(det_by_frame, frame_id)

            t0 = time.perf_counter()
            tracks_out = tracker.update(all_dets, dummy_img)
            tracks = byte_style_to_out(tracks_out)
            t1 = time.perf_counter()
            seq_time += (t1 - t0)

            write_mot_output(f, tracks, frame_id)
    return seqlen, seq_time

if __name__ == "__main__":
    sequences = find_sequences(GT_ROOT)
    tot_frames = 0
    tot_time = 0
    for seq in sequences:
        seqlen, seq_time = run_tracking(seq)
        tot_frames += seqlen
        tot_time += seq_time
        fps = seqlen / seq_time if seq_time > 0 else 0
        print(f"DONE {seq.name}, Time: {seq_time:.2f}s, FPS: {fps:.2f}")
    overall_fps = tot_frames / tot_time if tot_time > 0 else 0
    print(f"Overall Time {tot_time:.2f}s, Overall FPS {overall_fps:.2f}")




