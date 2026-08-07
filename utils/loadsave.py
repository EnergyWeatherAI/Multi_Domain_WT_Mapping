from torch import save, load
import json
from models import base_models, stargan_model

###
# Helper functions to save/load statistics, NBMs, or domain mapping models
###

def save_stats(stats, stats_path): 
    '''
    Given a dictionary of training statistics, save it in the specified path as a json file.
    '''
    with open(stats_path, 'w+') as f: json.dump({k: list(stats[k]) for k in stats.keys()}, f, indent=4)



def load_stats(stats_path):
    '''
    Given a specified path containing a json file of saved normalization statistics, load it into a dictionary and return it
    '''
    with open(stats_path) as json_file: 
        stats = json.load(json_file)
    return stats


def save_checkpoint(path, model, optimizer, epoch, tr_loss):
    '''
    Saves a model state with state dictionary and additional info to the specified path.
    '''
    save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
            'loss': tr_loss,
            }, path)


def load_pretrained_NBM(model_save_path, model_in_ch, device, return_threshold = False):
    '''
    Initializes a new ae-NBM and loads the weights from the state dict in the specified path.
    If set, this also calculates and loads the NBM's threshold, according to the specified formula.
    Returns the model in eval mode and transferred to the device.
    '''
    ae_model = base_models.base_AE_CNN(in_channels=model_in_ch)
    model_checkpoint = load(model_save_path.joinpath("nbm.pt"), map_location=device)
    ae_model.load_state_dict(model_checkpoint["model_state_dict"])
    ae_model = ae_model.to(device)
    ae_model = ae_model.eval();
    if not return_threshold: 
        return ae_model
    else:
        # calculate & load corresponding threshold based on the normal val. data reconstruction errors
        with open(model_save_path.joinpath("normal_data_results.json")) as json_file: results = json.load(json_file)
        threshold = results["val_N"]["mae"]["q3"] + ( 3 *  (results["val_N"]["mae"]["q3"] - results["val_N"]["mae"]["q1"]))
        return ae_model, threshold



def load_pretrained_gen_OPT(model_save_path, model_in_ch, n_wts, device):
    '''
    Initializes a new generator (mapped) and loads the weights from the specified path.
    Returns the model in eval mode and transferred to the device.
    '''
    # generator
    gen = stargan_model.GeneratorTCN(n_feat = model_in_ch, n_cls=n_wts)
    model_checkpoint = load(model_save_path.joinpath(f"gen_best.pt"), map_location=device)
    gen.load_state_dict(model_checkpoint["model_state_dict"])
    gen = gen.to(device)
    gen = gen.eval();
    return gen