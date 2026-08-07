import numpy as np
import pathlib, argparse
import torch

from evaluation import mapping_eval
from evaluation.mapping_model_selection import get_all_deltaT
from utils.data_utils import wt_id_to_label
from utils import loading_utils, setup_utils
from utils.loadsave import load_stats, load_pretrained_gen_OPT
import json


#####################################
#   Main script to evaluate a trained domain mapping model (mapper).
#   Calculates the model's proxy metric (source WT validation sets) and similarity performance
#   of all directions, with key summaries for source-to-source F1 and target-to-sources F1.
#   evaluate_mapping.py must be provided with 
#   Example [python] evaluate_mapping.py -SETUP_UP=1 -SCARCITY="2w" -CUDA_IDX=0 
#   By default, all various model configurations that match the setup and scarcity are evaluated, unless they already have been.
#   NOTE: The assumption is that all various model configurations (different lambdas) follow the same normalization procedure.
#   The script runs the corresponding evaluation script and automatically saves the results in the /results/ folder.
######################################

# ---- CLI PARSING -----
parser = argparse.ArgumentParser()
parser.add_argument('-CUDA_IDX', help='GPU CUDA index, exclude for cpu training', default = -1)
parser.add_argument('-SETUP_ID', type=int, help='defines which setup (which source WTs and target WT), 1-6')
parser.add_argument('-SCARCITY', type=str, help='Set the scarcity scenario as string for 1w, 2w, 3w, 1m, 6w, 2m. \
                                            Do not provide anything to train with full training sets', default=None)

args = parser.parse_args()
# ------------------------


import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.CUDA_IDX)


def main(args):
    # get site name and WT id for the N source WTs and the last target WT defined by setups in the setup_utils
    SITES, WT_IDS = setup_utils.get_setup_sites_ids(args.SETUP_ID)

    # defines the training data scarcity scenario (only) for the last WT; the data-scarce target WT
    TARGET_SCARCITY = "" if args.SCARCITY is None else f"{args.SCARCITY}"

    np.random.seed(7)
    torch.manual_seed(7)

    # cuda
    #device = torch.device(f'cuda:{args.CUDA_IDX}' if torch.cuda.is_available() else 'cpu')
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("medium")

    ################################
    # DATA SPECIFICATIONS & LOADING
    ################################
    FOLDER_NAME = f"SETUP_{args.SETUP_ID}_{TARGET_SCARCITY}"

    DATA_PATH = pathlib.Path.cwd().joinpath("dataset") # must contain the WT's raw SCADA .csv file
    META_CSV_PATH = DATA_PATH.joinpath("META.csv")

    N_WTS = len(SITES)
    WT_LABELS = [wt_id_to_label(i, N_WTS) for i in range(0, N_WTS)]
    SCARCITIES = [None for i in range(0, N_WTS)]
    SCARCITIES[-1] = TARGET_SCARCITY

    x_features = ["Power_min", "Power_avg", "Power_max", "WindSpeed_min", "WindSpeed_avg", "WindSpeed_max", "RotorSpeed_min", "RotorSpeed_avg", "RotorSpeed_max"] + ["StatorTemp1", "RotorTemp1"]
    # dictionary like 0: site: site_name, wt_id: wt_id, wt_label: wt_label etc. 
    wt_dicts = [dict(zip(["site", "WT_ID", "scarcity", "wt_label"], [SITES[i], WT_IDS[i], SCARCITIES[i], WT_LABELS[i]])) for i in range(0, N_WTS)]
    wt_infos = dict(zip(list(range(0, N_WTS)), wt_dicts))

    print("Loading training and test sets...")
    base_config = {"x_features": x_features, "seq_len": 72, "val_size": 0.30, "test_size": 0.30, "bs": 256}
    scada_trsets = loading_utils.load_many_trvalsets(base_config, wt_infos, DATA_PATH, META_CSV_PATH) # takes care of norm.

    p = pathlib.Path.cwd().joinpath("saves", "map", FOLDER_NAME)
    MODEL_CONFIGS = [f.name for f in p.iterdir() if f.is_dir()]

    # test sets; load the training statistics *from the first* configuration
    mapping_save_path = p.joinpath(MODEL_CONFIGS[0])
    TR_STATS = load_stats(mapping_save_path.joinpath("tr_stats.json"))


    # load the test sets from each setup WT
    base_config = {"x_features": x_features, "seq_len": 72, "test_size": 0.30, 
                   "bs": 1024, "tr_stats": TR_STATS}
    scada_testsets = loading_utils.load_many_testsets(base_config, wt_infos, DATA_PATH)
    print("...finished loading data....")

    
    # load all pretrained NBMs and corresponding (normal-only) validation and (potentially-anomalous) test set ground truths for evaluation
    repr_NBMs = loading_utils.batch_load_NBMs(SITES, WT_IDS, x_features, device)
    repr_results_val = loading_utils.get_repr_scores_CUDA(repr_NBMs, scada_trsets, device, "val")
    repr_results_test = loading_utils.get_repr_scores_CUDA(repr_NBMs, scada_testsets, device, "test")


    # evaluate each config (that hasn't been evaluated before):
    for i, config in enumerate(MODEL_CONFIGS):
        
        # where to store the results of that config
        results_path = pathlib.Path.cwd().joinpath("results", f"mapping_setup_{args.SETUP_ID}_{TARGET_SCARCITY}", config)
        results_path.mkdir(parents=True, exist_ok=True)
     
        # update the path to match the current model config
        mapping_save_path = p.joinpath(config)



        # skip evaluation if already evaluated
        filecheck = results_path.joinpath("mapping_results.json")
        if filecheck.is_file():
            print("Configuration already evaluated, skipping. \n -----")
            continue

        final_results = {}
        print(f"Evaluating config: {config} ({(i+1)} / {len(MODEL_CONFIGS)}).")
        
        # load the corresponding generator (discriminator not needed)
        gen = load_pretrained_gen_OPT(mapping_save_path, model_in_ch=len(x_features), n_wts=N_WTS,
                                                device=device)


        # get mapped anomaly scores by mapping *normal* validation data across all source WTs 
        # dictionary with scores @ m{origin_id}{destination_id} for origin->destination mapping
        mapped_scores_val = mapping_eval.get_mapped_anomaly_scores(gen, repr_NBMs, scada_trsets, device, 
                                    selected_set = "val", source_WTs_only=True)
        
        # calculate the deltaT for the model selection evaluation proxy metric
        delta_Ts = get_all_deltaT(mapped_scores_val, repr_results_val)


        # evaluate test set performance for source to source and target to source
        # first, get mapped anomaly scores on test set data
        mapped_scores_test = mapping_eval.get_mapped_anomaly_scores(gen, repr_NBMs, scada_testsets, device, 
                                    selected_set = "test")

        # calculate similarity metrics, especially average F1s
        f1s_test = mapping_eval.get_avg_F1_score(mapped_scores_test, repr_results_test, repr_NBMs, ignore_scarce_destination = True)

        # save everything into a big dictionary
        final_results["delta_Ts"] = delta_Ts
        final_results["f1s_test"] = f1s_test

        # print out key results
        verbose = True 
        if verbose:
            print(f"Average delta T (sources<->sources validation data): {delta_Ts['mean']}")
            print(f"Average F1 for sources<->sources: {f1s_test['src_to_src_summary']['avg_F1']}")
            print(f"Average F1 for target->sources: {f1s_test['scarce_summary']['avg_F1']}")
            print(f"Ensemble F1s for target->sources: {f1s_test['scarce_summary']['ensemble_F1']}")
            print("----\n")

        # for that configuration, save dictionary as final results in a json
        with open(results_path.joinpath("mapping_results.json"), "w") as f: json.dump(final_results, f, indent=4, cls=NumpyEncoder)
        print("---Finished evaluation---")

class NumpyEncoder(json.JSONEncoder):
    # helper class to handle numpy data in the dictionaries for the conversion to json
    def default(self, obj): 
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

if __name__ == "__main__":
    main(args)