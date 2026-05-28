from pathlib import Path
from ultralytics import YOLO
def main():
    home = Path.home()
    data_yaml = home / "dataset" / "Train_YOLO" / "data.yaml"
    results_folder = home / "train" / "results"
    model = YOLO("yolo26s.pt")

    model.train(
        data=str(data_yaml),
        project=str(results_folder),
        epochs=80,
        imgsz=1440,
        optimizer = "SGD",
        lr0=0.001,
        momentum=0.9,
        weight_decay=0.0005,
        warmpup_epochs=1,
        cos_lr = True,
        batch = 6,
        mosaic = 1.0,
        mixup = 1.0,
        close_mosaic = 10,
        name="ByteTrack1"
    )

if __name__ == "__main__":
    main()