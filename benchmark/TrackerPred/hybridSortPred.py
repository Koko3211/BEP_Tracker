import sys
import time
from pathlib import Path
from types import SimpleNamespace
from benchmark.utils.mot_out import write_mot_output, byte_style_to_out
from benchmark.utils.seqinfo import read_seqinfo, find_sequences
from benchmark.utils.load_det import load_detections, get_frame_dets

# NOTE use SUT env
# Path setup
BEP_ROOT = Path(__file__).resolve().parents[2]

#import Tracker
from HybridSORT.trackers.hybrid_sort_tracker.hybrid_sort import Hybrid_Sort, KalmanBoxTracker

# Path setup for output and input
TRACKER_NAME = "hybridsort"
SPLIT = "MFT-test"

METRICS = BEP_ROOT / "trackeval"
GT_ROOT = METRICS / "gt" / SPLIT
OUT_DIR = METRICS / "trackers" / SPLIT / TRACKER_NAME / "data"
DET_ROOT = BEP_ROOT / "benchmark" / "dets" / SPLIT

def run_tracking(seq_dir):
    args = SimpleNamespace(track_thresh = 0.5, TCM_first_step = False, TCM_byte_step= False,
                            TCM_byte_step_weight = 1, mot20= False, ablation = False)
    tracker = Hybrid_Sort(args, det_thresh= args.track_thresh, max_age=40, min_hits=3,
                           iou_threshold=0.3, delta_t=3, asso_func="Height_Modulated_IoU", inertia=0.2, use_byte=True)

    width, height, seqlen = read_seqinfo(seq_dir)

    img_info = [height, width]
    img_size = (height, width)

    det_file = DET_ROOT / f"{seq_dir.name}.txt"
    det_by_frame = load_detections(det_file)
    
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    pred_file = OUT_DIR / f"{seq_dir.name}.txt"

    #Initialize time
    seq_time = 0

    with pred_file.open("w", encoding="utf-8") as f:
        for frame_id in range(1, seqlen + 1):
            all_dets = get_frame_dets(det_by_frame, frame_id)

            t0 = time.perf_counter()
            tracks = tracker.update(all_dets, img_info, img_size)
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




