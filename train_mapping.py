import numpy as np
import pathlib, argparse, gc
import torch

from models import stargan_model
from trainers import mapping_trainer
from utils.loadsave import save_stats
from utils.data_utils import wt_id_to_label
from utils import loading_utils, setup_utils


#####################################
#   Main script to train a StarGAN-based domain mapping model for a specified setup.
#   train_mapping.py must be provided with the follwing information:
#   Example [python] train_mapping.py -SETUP_ID=1 -SCARCITY="2w" -CUDA_IDX=0
#
#   Further settings are determined by the configuration dictionary within this script.
#   The script runs the corresponding training script and automatically saves the final generator in the /saves/map/ folder.
######################################

# ---- CLI PARSING -----
parser = argparse.ArgumentParser()
parser.add_argument('-CUDA_IDX', help='GPU CUDA index', default = 0)
parser.add_argument('-SETUP_ID', type=int, help='defines which setup (which source WTs and target WT), 1-6')
parser.add_argument('-SCARCITY', type=str, help='Set the scarcity scenario as string for 1w, 2w, 3w, 1m, 6w, 2m. \
                                            Do not provide anything to train with full training sets', default=None)

parser.add_argument('-CYC', type=float, default=2.5)
parser.add_argument('-ZERO', type=float, default=2.0)
parser.add_argument('-MAX', type=float, default=0.3)
parser.add_argument('-CLS_G', type=float, default=0.1)
parser.add_argument('-CLS_D', type=float, default=1.0)



args = parser.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.CUDA_IDX)


def main(args):
    # get site name and WT id for the N source WTs and the last target WT defined by setups in the setup_utils
    SITES, WT_IDS = setup_utils.get_setup_sites_ids(args.SETUP_ID)

    # defines the training data scarcity scenario (only) for the last WT; the data-scarce target WT
    TARGET_SCARCITY = "" if args.SCARCITY is None else f"{args.SCARCITY}"

    # hyper-parameters for the loss weights (cycle-consistency loss, zero-loss, max-loss, and class loss weights)
    LAMBDAS = {"cyc": args.CYC, "zero": args.ZERO, "max": args.MAX, "cls_G":args.CLS_G, "cls_D":args.CLS_D}        

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


    print("Loading training sets...")
    base_config = {"x_features": x_features, "seq_len": 72, "val_size": 0.30, "test_size": 0.30, "bs": 32}
    scada_datasets = loading_utils.load_many_trvalsets(base_config, wt_infos, DATA_PATH, META_CSV_PATH) 

    # preparing saving
    save_name = f"c_{LAMBDAS['cyc']}_m_{LAMBDAS['max']}_z_{LAMBDAS['zero']}_clsG_{LAMBDAS['cls_G']}_clsD_{LAMBDAS['cls_D']}"
    pathlib.Path.cwd().joinpath("saves", "map", FOLDER_NAME, save_name).mkdir(parents=True, exist_ok=True)
    save_path = pathlib.Path.cwd().joinpath("saves", "map", FOLDER_NAME, save_name)

    ####################################
    #     DOMAIN MAPPING TRAINING      #
    ####################################
    print("Preparing models...")
    gen = stargan_model.GeneratorTCN(n_feat = len(x_features), n_cls = N_WTS).to(device)
    disc = stargan_model.Discriminator(n_feat=len(x_features), n_cls = N_WTS).to(device)

    # optimizers
    opt_G =  torch.optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_D =  torch.optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.999)) 

    # the normalization stats, set by the training set, are saved together with the model for later evaluation
    tr_stats = scada_datasets[0]["data"]["tr"]["stats"]
    save_stats(tr_stats, pathlib.Path.cwd().joinpath(save_path, "tr_stats.json"))


    # Early stopping is performed once the reconstruction error by source NBMs rises on the scarce, normal target validation data 
    # load the source NBMs, and the stats for the norm changer
    repr_NBMs = loading_utils.batch_load_NBMs(SITES, WT_IDS, x_features, device)
    mapping_norm_stats = scada_datasets[0]["norm_stats"]

    # trainer configuration (see trainers/mapping_trainer)
    trainer_config = {
        "n_wts": N_WTS,
        "lambdas": LAMBDAS, # hyperparameters for loss weighting
        "device": device, 
        "save_dir": save_path, 
        "max_powers" : [scada_datasets[key]["rated_pwr_normed"] for key in scada_datasets.keys()], # for the rated power loss 

        # required for the early stopping criterion (mapping target validation data for source NBM evaluation)
        "repr_NBMs": repr_NBMs,
        "mapping_norm_stats": mapping_norm_stats,
        }

    mytrainer = mapping_trainer.Trainer(trainer_config)
        
    ###########
    # TRAINING#
    ###########
    mapping_network = {"gen": gen, "disc": disc}
    tr_dataloaders = [scada_datasets[key]["torch_dataloaders"] for key in scada_datasets.keys()]
    # for early stopping, also provide the validation (normal data) of the target WT
    target_val_dataloader = scada_datasets[N_WTS-1]["torch_dataloaders"]["val"]
    opts = {"opt_G": opt_G, "opt_D": opt_D}

    print("...Training mapping...")
    _ = mytrainer.train(max_gen_iter = 12501, models=mapping_network, tr_dataloaders=tr_dataloaders, 
                                        target_val_dataloader = target_val_dataloader, optimizers=opts)


    # clean up
    del gen, disc, scada_datasets
    gc.collect()
    if device !="cpu": torch.cuda.empty_cache()
    print("\n\n\n-----FINISHED---------")

if __name__ == "__main__":
    main(args)