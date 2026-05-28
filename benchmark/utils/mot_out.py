import numpy as np

def write_mot_output(f, tracks, frame_id):
    for track in tracks:
        x1, y1, x2, y2, id = track
        #Convert to MOTChallenge format
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        f.write(f"{frame_id},{id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{1},-1,-1,-1\n")

def byte_style_to_out(out_tracks):
    if out_tracks is None or len(out_tracks) == 0:
        return np.empty((0, 5), dtype=np.float32)

    rows = []

    for track in out_tracks:
        x1, y1, x2, y2 = track.tlbr
        track_id = track.track_id
        rows.append([x1, y1, x2, y2, track_id])

    return np.asarray(rows, dtype=np.float32)