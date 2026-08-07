import numpy as np 
import pathlib 

from utils.loadsave import load_stats, load_pretrained_NBM
from utils.data_utils import scarcity_to_n_seq, get_norm_changer_CUDA
from evaluation import nbm_eval

from data.scada_trainingset import SCADA_Trainingset
from data.scada_testset import SCADA_Testset

###
# Helper functions to load several SCADA_Trainingsets, Testsets, NBMs, models, and anomaly scores.
###


def load_many_trvalsets(base_config, wt_infos, data_path, meta_path, stats_to_override = None):
    '''
    Loads several instances of SCADA_Trainingsets according to a configuration and returns them as a dict.
    
    Args:
        base_config (dict): Defines the base configuration of a single SCADA_Trainingset instance (see therein)
        wt_infos (dict): 
            A dictionary defining the setup / members of a setup for which to load the data for.
            A dictionary defining the distinguishing information of the WTs for which the trainingsets are created.
            Of form:  wt_infos{0} = {'site': <site_name>, 'WT_ID' <id_of_wt_in_farm>, 
                                            'scarcity': <WT's training scarcity setting, e.g., '2w'>, 
                                                    'wt_label': <one_hot_encoded_label>}
            See train_domainmapping.py for an application example. 

            data_path (str), meta_csv_path (str): See SCADA_Trainingset.py
            stats_to_override: 
            If wanted, specific stats can be specified here to norm *ALL* trainingsets.
            NOTE: **By default, the *first* WT's statistics are used to normalize all other trainingsets.**
    '''

    scada_datasets = {}

    # load a separate trainingset instance for each WT
    for i, wt in enumerate(wt_infos.keys()):
        wt_site = wt_infos[wt]["site"]

        # update config to reflect WT-specific information and data scarcity scenario 
        base_config.update({"SITE_NAME": wt_site, "limit_tr_to": scarcity_to_n_seq(wt_infos[wt]["scarcity"])})
        csvpath = data_path.joinpath(wt_site, f"{wt_site}_WT_0{wt_infos[wt]["WT_ID"]}.csv")

        # if stats are provided, override default behavior 
        if stats_to_override is not None: base_config.update({"overwrite_stats": stats_to_override})

        # load & add dataset to main dictionary
        scada_datasets[wt] = SCADA_Trainingset(base_config, wt_infos[wt]["wt_label"], csvpath, meta_path).get_data()

        # By default, the stats from the first WT are stored and used subsequently to normalize all other WTs
        if i == 0 and stats_to_override is None: 
            stats = scada_datasets[0]["data"]["tr"]["stats"]
            base_config.update({"overwrite_stats": stats})

    return scada_datasets



def load_many_testsets(base_config, wt_infos, data_path):
    ''' See load_many_trvalsets; but with the Testset instances
        NOTE: The training set statistics used to create the corresponding trainingsets should be set
         in the config dictionary for Testsets (scada_testset.py).
    ''' 

    scada_testsets = {}
    for i, wt in enumerate(wt_infos.keys()):
        wt_site = wt_infos[wt]["site"] # used to build the datafile/csv path

        # update config to reflect WT-specific information and data scarcity scenario 
        csvpath = data_path.joinpath(wt_site, f"{wt_site}_WT_0{wt_infos[wt]["WT_ID"]}.csv")

        # load & add dataset to main dictionary
        scada_testsets[wt] = SCADA_Testset(base_config, wt_infos[wt]["wt_label"], csvpath).get_data()

    return scada_testsets



def batch_load_NBMs(SITES, SITE_IDS, x_features, device):
    '''
    Loads the pretrained NBM for all WTs.
    Returns them in a dictionary containing the model, the calculated threshold, and the normalization statistics.
    
    Args:
        SITES (list): A list of site names from which site to load the NBMs from.
        SITE_IDS (list): Corresponding WT id of which WT to include from each site.
        x_features (list): List of x features used during training; required for model init
        device: CUDA device or cpu
    '''

    N_WTS = len(SITES)
    repr_NBMs = {}

    for i in range(0, N_WTS):
        rsavepath = pathlib.Path.cwd().joinpath("saves", "NBM", f"{SITES[i]}_WT_0{SITE_IDS[i]}")
        # load pretrained NBM with threshold
        NBM, TH = load_pretrained_NBM(rsavepath, len(x_features), device, return_threshold=True)
        stats = load_stats(rsavepath.joinpath("stats.json"))
        
        repr_NBMs[i] = {
            "NBM": NBM,
            "TH": TH,
            "stats": stats
        }
    return repr_NBMs


def get_repr_scores_CUDA(repr_NBMs, datasets, device, set_selection = "test"):
    '''
    Returns a dictionary containing the "ground truth" anomaly scores (reconstruction errors), obtained
    by evaluating the specified dataset on the pretrained representatitve NBMs of each WT.
    
    Args:
        repr_NBMs (dict): A dictionary containign the pretrained NBM, the threshold, and normalization stats.
        datasets (dict/list):  of SCADA_Training- or SCADA_Testsets
        device: CUDA device or cpu
        set_selection: Defines which set to evaluate to get the anomaly scores (str; 'tr', 'val', or 'test')
    '''

    repr_results = {}
    N_WTS = len(repr_NBMs)

    for i in range(0, N_WTS):
        # load that WT's pretrained, representative NBM
        nbm_dict = repr_NBMs[i]
        nbm, th, stats = nbm_dict["NBM"], nbm_dict["TH"], nbm_dict["stats"]
        
        # get a norm changer to evaluate data using the same normalization as during NBM training
        norm_changer_f = get_norm_changer_CUDA(target_stats=stats, unnorm_stats=datasets[0]['norm_stats'], 
                                                         device=device)

        # obtain reconstruction errors / anomaly scores
        wt_ds = datasets[i]
        
        scores =  np.asarray(nbm_eval.get_reconstr_errors(nbm, wt_ds["torch_dataloaders"][set_selection], 
                                                                    device, norm_change_f=norm_changer_f)["mae"])
        timestamps, incident_flags = wt_ds["data"][set_selection]["last_timestamps"], wt_ds["data"][set_selection]["incident_flags"]

        repr_results[i] = {"scores": scores, "timestamps": timestamps, "incident_flags": incident_flags}
    return repr_results
    