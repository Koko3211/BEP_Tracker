from pathlib import Path
from configparser import ConfigParser

def read_seqinfo(seq_dir: Path): 
    """
    Reads seqinfo.ini 
    Returns width, height, num_frames
    """
    seqinfo_file = seq_dir / "seqinfo.ini"
    if not seqinfo_file.exists():
        raise FileNotFoundError(f"seqinfo.ini not found in {seq_dir}")
    
    config = ConfigParser()
    config.read(seqinfo_file)

    width = int(config["Sequence"]["imWidth"])
    height = int(config["Sequence"]["imHeight"])
    seqlen = int(config["Sequence"]["seqLength"])

    return width, height, seqlen

def find_sequences(gt_dir: Path):
    sequences = [p for p in gt_dir.iterdir() if p.is_dir() and (p / "seqinfo.ini").exists()]
    return sequences