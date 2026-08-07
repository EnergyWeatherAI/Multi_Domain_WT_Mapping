import numpy as np 
import torch 
from evaluation.mapping_eval import map_and_reconstruct
from utils.data_utils import wt_id_to_label, get_norm_changer_CUDA



###
# Helper class and function to perform early stopping when training the domain mapping network.
###
@torch.no_grad()
def mapped_target_avg_score(gen, repr_NBMs, target_val_dl,
                                         mapping_norm_stats, scarce_id, device):
    
    '''
    Calculates the average reconstruction error (anomaly score) of mapped target WT validation data.
    Returns a float for the average obtained across all anomaly scores from all source WTs.

    Args:
        gen: The current domain mapping generator candidate (should be EMA)
        repr_NBMs (dict): All pretrained NBMs including stats
        target_val_dl (torch dataloader): Dataloader for the validation (normal only!) data of the target WT
        mapping_norm_stats (dict): 
            Dictionary defining which statistics were used to normalize the training data for the mapping.
            Used to create a norm changer, as the source NBMs require different normalization schemes.
        scarce_id (int): The id of the WT in the setup (usually n_wts-1 for N_WTS)
        device: CUDA device or cpu
    '''
    
    gen.eval()
    n_wts = len(repr_NBMs)
    source_ids = [j for j in range(n_wts) if j != scarce_id]

    per_direction_means = []

    # map the target WT validation data (origin) to every other source WT (destination)
    for j in source_ids:
        dest_nbm = repr_NBMs[j]["NBM"]
        dest_stats = repr_NBMs[j]["stats"]
        dest_label = wt_id_to_label(j, n_wts)

        # get norm changer to change the normalization from domain mapping to destination NBM
        norm_changer = get_norm_changer_CUDA(target_stats=dest_stats, 
                                                unnorm_stats=mapping_norm_stats, 
                                                    device=device)
        
        # get reconstruction errors (anomaly scores) evaluated on the source (destination) NBM 
        mae = map_and_reconstruct(gen, dest_nbm, target_val_dl, 
                                        dest_label, device, norm_changer)["mae"]
        
        # average across all
        per_direction_means.append(float(np.mean(mae)))
    
    print("Early Stopping Criterion Value:", float(np.mean(per_direction_means)))

    return float(np.mean(per_direction_means))


class MappingEarlyStopper:
    '''
    Early stopper class to help stop training once the average validation loss has not 
    reduced in value after patience steps. 
    '''
    def __init__(self, warmup_iters=500, patience=4, min_delta=0.0):

        self.warmup_iters = warmup_iters 
        self.patience = patience 
        self.min_delta = min_delta 

        self.best = float("inf")
        self.best_iter = None 
        self.counter = 0
    
    def update(self, iteration, score):

        if iteration < self.warmup_iters:
            return {"is_best": False, "stop": False, "score": score}
        
        is_best, stop = False, False 
        if score < self.best - self.min_delta:
            self.best = score
            self.best_iter = iteration 
            self.counter = 0 
            is_best = True 
        
        else:
            self.counter +=1 
            if self.counter >= self.patience: stop = True 

        return {"is_best": is_best, "stop": stop, "score": score}