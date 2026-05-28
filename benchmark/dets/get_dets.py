from pathlib import Path
import cv2
from ultralytics import YOLO

# Path setup
BEP_ROOT = Path(__file__).resolve().parents[2]
# Change test to val if needed
SPLIT = "MFT-test"
IMG_DIR = BEP_ROOT / "OUTDIR" / "trackeval" / "gt" / SPLIT
DET_OUT_DIR = BEP_ROOT / "benchmark" / "dets" / SPLIT
MODEL_PATH = BEP_ROOT / "UltralyticsYolo" / "YOLO_trained" / "YoloTrained26s.pt"

CONF = 0.05
IOU = 0.5
IMGSZ = 1440
MODEL = YOLO(MODEL_PATH)

def find_sequences(test_root: Path):
    sequences = [p for p in test_root.iterdir() if p.is_dir() and (p / "img1").exists()]
    return sequences

def find_frames(img_dir: Path):
    frames = [p for p in img_dir.iterdir() if p.is_file() and p.suffix == ".jpg"]
    return frames

def get_detections(seq_path: Path):
    img_dir = seq_path / "img1"
    frames = find_frames(img_dir)
    
    DET_OUT_DIR.mkdir(parents=True, exist_ok=True)
    det_file = DET_OUT_DIR / f"{seq_path.name}.txt"

    with det_file.open("w", encoding="utf-8") as f:
        for img_path in frames:
            frame = cv2.imread(str(img_path))

            frame_id = int(img_path.stem)
            result = MODEL(frame, imgsz=IMGSZ, conf=CONF, iou=IOU, single_cls=True, device=0, verbose=False)[0]
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            for box, conf in zip(xyxy, confs):
                x1, y1, x2, y2 = box
                f.write(f"{frame_id},{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f},{conf:.4f}\n")

if __name__ == "__main__":
    sequences = find_sequences(IMG_DIR)
    for seq in sequences:
        get_detections(seq)
        print(f"DONE {seq.name}")