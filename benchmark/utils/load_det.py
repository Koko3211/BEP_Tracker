from pathlib import Path
import numpy as np

def load_detections(det_file):
    """
    Load detections from a text file in the format:
    frame_id,x1,y1,x2,y2,confidence
    Returns a dictionary mapping frame_id to a list of detections, where each detection is a tuple (x1, y1, x2, y2, confidence).
    """
    dets_by_frame = {}

    with det_file.open("r") as f:
        for line in f:
            parts = line.split(",")

            frame_id, x1, y1, x2, y2, conf = parts
            frame_id = int(frame_id)
            x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
            conf = float(conf)

            if frame_id not in dets_by_frame:
                dets_by_frame[frame_id] = []
            dets_by_frame[frame_id].append((x1, y1, x2, y2, conf))
    
    dets_by_frame = {frame_id: np.array(dets) for frame_id, dets in dets_by_frame.items()}
    return dets_by_frame

def get_frame_dets(dets_by_frame, frame_id):
    return dets_by_frame.get(frame_id,np.empty((0, 5), dtype=np.float32),)

def filter_conf(dets, conf_thresh):
    return dets[dets[:, 4] >= conf_thresh]