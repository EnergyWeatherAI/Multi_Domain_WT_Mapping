from torch.nn.functional import one_hot
from data.torch_data import get_normalize_f, get_unnormalize_f
import numpy as np
from torch import tensor, LongTensor, no_grad, from_numpy, float32

###
# Helper functions
###
def scarcity_to_n_seq(scenario):
    '''
    Converts a data scarcity scenario (str; such as '1w' or '1m') into the corresponding amount of SCADA samples.
    '''
    convert_dict = { "1w": 1008, "2w": 2016, "3w": 3024,"1m": 4032, "6w": 6048, "2m": 8064,"3m": 12096, "None": None, None:None }
    return convert_dict[scenario]

def wt_id_to_label(wt_id, n_wts):
    '''
    Given a j-wth WT (int) within within a setup of N wts (int; n_wts),
    this returns a torch one-hot encoded label vector [0, ..., 1 (j), ..., 0]
    '''
    with no_grad():
        return one_hot(tensor(wt_id).type(LongTensor), n_wts)

def get_normalization(stats):
    '''
    Returns a torch-adapted transformation function to normalize (norm_fX) and de-normalize (unnorm_fX). To be provided for dataset instances.
    '''
    mins, maxs = from_numpy(np.asarray(stats["X_mins"])).type(float32), from_numpy(np.asarray(stats["X_maxs"])).type(float32)
    norm_fX = get_normalize_f(mins, maxs, b=1, a=-1)
    unnorm_fX = get_unnormalize_f(mins, maxs, b=1, a=-1)
    return norm_fX, unnorm_fX

def get_normalization_CUDA(stats, device):
    '''
    Returns a torch-adapted transformation function to normalize (norm_fX) and de-normalize (unnorm_fX). To be provided for dataset instances.
    norm_fX(x) will normalize x, unnorm_fX(x) returns data to their original values when normalized with norm_fX.
    '''
    mins, maxs = from_numpy(np.asarray(stats["X_mins"]).copy()).type(float32).to(device), from_numpy(np.asarray(stats["X_maxs"]).copy()).type(float32).to(device)
    norm_fX = get_normalize_f(mins, maxs, b=1, a=-1)
    unnorm_fX = get_unnormalize_f(mins, maxs, b=1, a=-1)
    return norm_fX, unnorm_fX

def get_norm_changer(target_stats, unnorm_fX_origin):
    '''
    The domain mapping network might be trained with different normalization statistics than NBMs.
    This function returns a function that de-normalizes the data using the original unnorm_fX, 
    and normalizes it with a new normalization function created with the target_stats.
    '''
    new_norm_fX, _ = get_normalization(target_stats)
    class Norm_changer:
        def __init__(self):
            self.unnorm_fX_origin = unnorm_fX_origin 
            self.new_norm_fX = new_norm_fX
        def change_norm(self, batch):
            return new_norm_fX(self.unnorm_fX_origin(batch))
    return Norm_changer().change_norm


def get_norm_changer_CUDA(target_stats, unnorm_stats, device):
    # returns same function as above but can be performed faster on CUDA
    new_norm_fX, _ = get_normalization_CUDA(target_stats, device)
    _, new_unnorm_fX = get_normalization_CUDA(unnorm_stats, device)
    class Norm_changer:
        def __init__(self):
            self.unnorm_fX_origin = new_unnorm_fX 
            self.new_norm_fX = new_norm_fX
        def change_norm(self, batch):
            return new_norm_fX(self.unnorm_fX_origin(batch))
    return Norm_changer().change_norm




