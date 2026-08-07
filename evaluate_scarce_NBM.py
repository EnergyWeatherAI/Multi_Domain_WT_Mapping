import numpy as np
import pathlib, json, argparse
import pandas as pd
import torch

from data.scada_testset import SCADA_Testset
from evaluation import nbm_eval
from utils.loadsave import load_pretrained_NBM

#####################################
#   Evaluation script to evaluate the performance of an NBM trained on scarce data v. trained on representative data.
#   evaluate_models.py must be provided with WT and scarcity information:
#   Example [python] evaluate_models.py -SITE_NAME="farm1" -WT_ID=1 -SCARCITY="2w" 
#
#   The script evaluates the same test set of the (target) WT
#   The script stores the calculated nbm_results.csv in the respective results folder. 
######################################

# ---- CLI PARSING -----
parser = argparse.ArgumentParser()
parser.add_argument('-SITE_NAME', help='wind site name of the *target* WT, e.g., farm2')
parser.add_argument('-WT_ID', help='id of the target WT', default=1)
parser.add_argument('-SCARCITY', type=str, help='Set the *target WT* scarcity scenario in (1w, 2w, 3w, 1m, 6w, 2m)', default="2w")
parser.add_argument('-CUDA_IDX', help='GPU CUDA index, exclude for cpu training', default = -1)
args = parser.parse_args()

def main(args):
    np.random.seed(7)
    torch.manual_seed(7)
    device = torch.device(f'cuda:{args.CUDA_IDX}' if torch.cuda.is_available() else 'cpu')

    ################################
    # DATA SPECIFICATIONS & LOADING
    ################################
    SITE_NAME, WT_ID = args.SITE_NAME, args.WT_ID
    WT_NAME = f"{SITE_NAME}_WT_0{WT_ID}"

    DATA_PATH = pathlib.Path.cwd().joinpath("dataset") # must contain the WT's raw SCADA .csv file
    scada_csv_path = DATA_PATH.joinpath(SITE_NAME, f"{SITE_NAME}_WT_0{WT_ID}.csv")
    meta_csv_path = DATA_PATH.joinpath("META.csv")
    SCARCITY = args.SCARCITY


    ###
    # SCADA Test set configuration shared by both test sets (everything but the normalization statistics)
    ###
    shared_config = {
        "SITE_NAME": SITE_NAME,
        "x_features": ["Power_min", "Power_avg", "Power_max", "WindSpeed_min", "WindSpeed_avg", "WindSpeed_max", 
                            "RotorSpeed_min", "RotorSpeed_avg", "RotorSpeed_max"] + ["StatorTemp1", "RotorTemp1"],
        "seq_len": 72, # 72 samples within a sequence <-> 12h
        "test_size": 0.30, # will be the last 30% of data (i.e., is independent of the scarcity)
        "bs": 256, # batch size for test set
    }

    wt_label = np.array([0]) # NOTE the wt label is irrelevant for NBMs, only for the domain mapping


    ################################
    # DATA SPECIFICATIONS & LOADING
    ################################

    # NOTE: The representative target NBM was normalized according to the statistics of the *full* target WT training set 
    # set statistics accordingly in the configuration
    with open(pathlib.Path.cwd().joinpath("saves", "NBM", WT_NAME, f"stats.json")) as json_file: 
        stats_full_target = json.load(json_file)
    config_representative = {"tr_stats": stats_full_target}
    config_representative.update(shared_config)

    # get the corresponding *test set* with this normalization:
    scada_testset_repr = SCADA_Testset(config_representative,wt_label,scada_csv_path).get_data()
    test_dl_repr = scada_testset_repr["torch_dataloaders"]["test"]


    # NOTE: The *scarce* target NBM was normalized according to the statistics of the *scarce* target WT training set 
    with open(pathlib.Path.cwd().joinpath("saves", "NBM", f"{WT_NAME}_{SCARCITY}", f"stats.json")) as json_file: 
        stats_scarce = json.load(json_file)

    config_scarce = {"tr_stats": stats_scarce}
    config_scarce.update(shared_config)

    # get the corresponding *test set* with this normalization:
    scada_testset_scarce = SCADA_Testset(config_scarce,wt_label,scada_csv_path).get_data()
    test_dl_scarce = scada_testset_scarce["torch_dataloaders"]["test"]



    ##################################
    #       LOAD NBMs & MODELS       #
    ##################################

    # 1) (target WT) NBM trained on *full dataset* (ground truth) with its corresponding threshold
    model_save_path = pathlib.Path.cwd().joinpath("saves", "NBM", WT_NAME)
    nbm_repr, repr_TH = load_pretrained_NBM(model_save_path, 
                                            model_in_ch = len(shared_config["x_features"]), 
                                                device=device, return_threshold=True)



    # 2) TARGET DOMAIN: NBM trained on *scarce*, partial data
    model_save_path = pathlib.Path.cwd().joinpath("saves", "NBM", f"{WT_NAME}_{SCARCITY}")
    nbm_scarce, scarce_TH = load_pretrained_NBM(model_save_path, 
                                            model_in_ch = len(shared_config["x_features"]), 
                                                device=device, return_threshold=True)


    ######################################
    #    MODEL EVALUATION AND COMPARISON #
    ######################################
    results = {}

    # anomaly scores (reconstruction errors) for each sample in the test set
    # for i) the target WT NBM trained on its full, representative training data
    scores_repr =  np.asarray(nbm_eval.get_reconstr_errors(nbm_repr, test_dl_repr, device)["mae"])
    # convert to binary threshold exceedance (1 (positive) if score >= threshold, 0 (negative) else)
    scores_binary_repr = nbm_eval.binarize_anomaly_scores(scores_repr, repr_TH)

    # anomaly scores from the target WT NBM trained on scarce data
    scores_scarce =  np.asarray(nbm_eval.get_reconstr_errors(nbm_scarce, test_dl_scarce, device)["mae"])
    scores_binary_scarce = nbm_eval.binarize_anomaly_scores(scores_scarce, scarce_TH)
    
    # calculate the threshold score similarity between the binarized scarce & representative scores
    # (see nbm_eval)
    scarce_results = nbm_eval.threshold_similarity_performance(scores_binary_repr, scores_binary_scarce)
    results["scarce"] = scarce_results 



    # 3) PUT IT INTO A DATAFRAME
    results_df = pd.DataFrame.from_dict(results)
    # & save the df
    results_path = pathlib.Path.cwd().joinpath("results", f"{WT_NAME}_{SCARCITY}")
    results_path.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path.joinpath(f"{WT_NAME}_{SCARCITY}_nbm_results.csv"))
    
    print(results_df)

    # clean up
    if device !="cpu": torch.cuda.empty_cache()
    print("\n\n\n-----FINISHED---------")

if __name__ == "__main__":
    main(args)