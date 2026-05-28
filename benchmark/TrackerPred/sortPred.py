import time
from pathlib import Path
from tracker.sort import Sort, KalmanBoxTracker
from benchmark.utils.mot_out import write_mot_output
from benchmark.utils.seqinfo import read_seqinfo, find_sequences
from benchmark.utils.load_det import load_detections, get_frame_dets, filter_conf
import numpy as np

# Path setup
BEP_ROOT = Path(__file__).resolve().parents[2]

#Changeable
TRACKER_NAME = "sort"
SPLIT = "MFT-test"

METRICS = BEP_ROOT / "trackeval"
GT_ROOT = METRICS / "gt" / SPLIT
OUT_DIR = METRICS / "trackers" / SPLIT / TRACKER_NAME / "data"
DET_ROOT = BEP_ROOT / "benchmark" / "dets" / SPLIT
CONF = 0.3
def run_tracking(seq_dir):
    #Reset SORT tracker count for each sequence
    KalmanBoxTracker.count = 0

    tracker= Sort(max_age=60, min_hits=3, iou_threshold= 0.3)
    _,_,seqlen = read_seqinfo(seq_dir)

    det_file = DET_ROOT / f"{seq_dir.name}.txt"
    det_by_frame = load_detections(det_file)
    
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    pred_file = OUT_DIR / f"{seq_dir.name}.txt"

    #Initialize time
    seq_time = 0

    with pred_file.open("w", encoding="utf-8") as f:
        for frame_id in range(1, seqlen + 1):
            all_dets = get_frame_dets(det_by_frame, frame_id)
            conf_dets = filter_conf(all_dets, CONF)
            t0 = time.perf_counter()
            tracks = tracker.update(conf_dets)
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
    print(f"Overall Time: {tot_time:.2f}s, Overall FPS: {overall_fps:.2f}")




