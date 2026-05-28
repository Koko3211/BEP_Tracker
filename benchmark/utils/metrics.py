from pathlib import Path
import trackeval

# Define trackers to evaluate, delete/add as needed
TRACKERS = ["our_sort", "botsort", "bytetrack", "hybridsort", "ocsort", "SUT", "sort"]
# set test / val
SPLIT = "test" 

def calculate_metrics(root_dir):
    eval_config = trackeval.Evaluator.get_default_eval_config()
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    
    eval_config["DISPLAY_LESS_PROGRESS"] = False
    eval_config["PRINT_RESULTS"] = True
    eval_config["PRINT_ONLY_COMBINED"] = False
    eval_config["OUTPUT_SUMMARY"] = True
    eval_config["OUTPUT_DETAILED"] = True
    eval_config["PLOT_CURVES"] = False

    dataset_config["GT_FOLDER"] = str(root_dir/ "gt")
    dataset_config["TRACKERS_FOLDER"] = str(root_dir/ "trackers")
    dataset_config["OUTPUT_FOLDER"] = str(root_dir / "output")
    dataset_config["BENCHMARK"] = "MFT"
    dataset_config["SPLIT_TO_EVAL"] = SPLIT
    dataset_config["TRACKERS_TO_EVAL"] = TRACKERS
    dataset_config["SEQMAP_FILE"] = str(root_dir / "gt" / "seqmaps" / f"seqmaps_{SPLIT}.txt")
    dataset_config["DO_PREPROC"] = False
    dataset_config["CLASSES_TO_EVAL"] = ["pedestrian"]
    dataset_config["PRINT_CONFIG"] = False

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]

    evaluator.evaluate(dataset_list, metrics_list)

if __name__ == "__main__":
    BEP_ROOT = Path(__file__).resolve().parents[2]
    ROOT = Path(BEP_ROOT / "trackeval")
    calculate_metrics(ROOT)