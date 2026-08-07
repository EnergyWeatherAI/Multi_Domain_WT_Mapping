import numpy as np 
import torch
from utils.data_utils import wt_id_to_label, get_norm_changer_CUDA
from evaluation.nbm_eval import threshold_similarity_performance

###############################
# *helper* functions to evaluate a trained stargan-based domain mapping model.
# The main domain mapping evaluation is in evaluate_mapping.py, refer there for further documentation.
###############################

def map_and_evaluate_samples(gen, repr_NBMs, datasets, origin_wt_id, destination_wt_id, 
                                                device, selected_set = "test"):
    '''
    Maps samples from an origin WT to a destination WT and evaluates them with the destination NBM.
    Returns mapped anomaly scores (and additional information such as corresponding timestamps and possible incident flags)

    Args:
        gen: the trained domain mapping generator
        repr_NBMs: A list of NBM dictionaries [i], should contain the pretrained NBM model for WT [i], with corresponding threshold, and training statistics
        datasets: A list or dictionary of SCADA datasets, either training or test sets. 
        origin_wt_id: The id i for the origin wt 
        destination_wt_id: id for which WT to map to (destination j)
        device: CUDA device
        selected_set: The scores will be mapped from samples using the datasets[i]["torch_dataloaders"][selected_set] data, so either "tr", "val", (given a trainingset class) or "test" (given a testset)
    '''
    
    # load stats *used for training that NBM*, pretrained NBM, and corresponding threshold from the destination WT
    dest_NBM, dest_TH, dest_stats = [repr_NBMs[destination_wt_id][key] 
                                                            for key in ["NBM", "TH", "stats"]]
    
    # load the stats *used during domain mapping training*
    mapping_norm_stats = datasets[0]['norm_stats']

    # get a norm changer, a torch-adapted function that will change the normalization of a sample batch to match the destination NBM's stats (see map_and_reconstruct function) 
    dest_norm_changer_f = get_norm_changer_CUDA(target_stats=dest_stats,unnorm_stats=mapping_norm_stats, device=device)


    # provide the (one-hot encoded) destination WT label (must match the domain mapping / training set)
    destination_label = wt_id_to_label(destination_wt_id, len(repr_NBMs))
    

    # map specified dataset samples and reconstruct/evaluate on destination domain and return anomaly scores
    mapped_scores_to_dest = np.asarray(map_and_reconstruct(gen, dest_NBM, datasets[origin_wt_id]["torch_dataloaders"][selected_set], destination_label, device, dest_norm_changer_f)["mae"])
    timestamps_from, incident_flags_from = datasets[origin_wt_id]["data"][selected_set]["last_timestamps"], datasets[origin_wt_id]["data"][selected_set]["incident_flags"]
    return {"mapped_scores": mapped_scores_to_dest, "timestamps": timestamps_from, "incident_flags": incident_flags_from}

def get_mapped_anomaly_scores(gen, repr_NBMs, scada_dataset, device, 
                                selected_set = "test", source_WTs_only = False):
    '''
    Returns a dictionary with mapped anomaly scores from specified or each origin to destination WT direction.
    For each origin WT, the specified samples (a set of the scada_dataset, e.g., 'val' set of SCADA_Trainingsets)
    are mapped to every destination WT and vice-versa (unless source_WTs_only specified).
    
    Args:
        gen: the trained domain mapping generator
        repr_NBMs: A list of NBM dictionaries [i], should contain the pretrained NBM model for WT [i], with corresponding threshold, and training statistics
        scada_dataset: A list or dictionary of SCADA datasets, either training or test sets. 
        device: CUDA device
        selected_set: The scores will be mapped from samples using the scada_dataset[i]["torch_dataloaders"][selected_set] data, so either "tr", "val", (given a trainingset class) or "test" (given a testset)
        source_WTs_only: If True, ignores the target WT (last WT in setup) in every direction, only source-to-source mappings.
    '''

    from_to_maps = {}
    N_WTS = len(repr_NBMs)

    for origin_wt_id in range(0, N_WTS):
        if source_WTs_only and origin_wt_id == (N_WTS - 1): continue

        from_to_scores = {}
        for destination_wt_id in [x for x in list(range(0, N_WTS)) if x != origin_wt_id]:
            if source_WTs_only and destination_wt_id == (N_WTS - 1): continue
            from_to_scores[destination_wt_id] = map_and_evaluate_samples(gen, repr_NBMs, scada_dataset, 
                                                    origin_wt_id, destination_wt_id, device, selected_set = selected_set)


        from_to_maps[origin_wt_id] = from_to_scores

    return from_to_maps

def map_and_reconstruct(mapper, nbm, dl, target_label, device, norm_change_f=None):
    '''
    Given a dataloader/samples from an origin WT, returns mapped anomaly scores (reconstruction errors)
    per sample. The destination WT is set by the target_label, and the mapped samples are evaluated using
    the destination WT's NBM. 
        
    Args:
        mapper (torch model): A generator/mapper of the domain mapping network to map in the direction of domain(provided batches) to the other domain. (Example: target domain)
        nbm (torch model): The trained NBM of the *other* domain, i.e., NOT from the domain of the batches. (Example: source domain NBM)
        dl (torch dataloader): Dataloader with batches from the domain to map  (Example: target domain data)
        target_label (torch-one-hot-encoded): Defines the destination WT with the matching target label corresponding to the training procedure.
        device (torch device): cuda device or cpu to perform the calculations on.
        norm_change_f (torch-adapted function): Converts the normalization of a batch with its specified statistics
    '''

    error_dict = {"mae": []}

    loader = iter(dl)
    tlbl_size = target_label.size(0)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            tgt_label_y = target_label.expand((x.size(0), tlbl_size)).to(device, non_blocking=True)
            x_mapped = mapper(x, tgt_label_y)

            if norm_change_f is not None:
                # change normalization to match the destination domain (pretrained NBM)
                x_mapped = norm_change_f(x_mapped)


            reconstr_mapped = nbm(x_mapped)
            mae_err = (torch.mean(torch.abs(x_mapped - reconstr_mapped), dim=[1,2])).tolist()
            error_dict["mae"].extend(mae_err)

    return error_dict

def get_avg_F1_score(all_mapped_anomaly_scores, ground_truths, repr_NBMs, ignore_scarce_destination = True):
    '''
    Returns a dictionary containing performance metrics for WT-to-WT mappings including overall summarized key metrics.
    
    Args:
        all_mapped_anomaly_scores (dict): 
            A dictionary containing mapped anomaly scores s.t. 
            d[origin][destination]['mapped_scores'] represents a mapping FROM origin TO destination
        ground_truths (list/dict): ground_truths[i] contains the ground truth anomaly scores for WT i (by representative NBM) 
        repr_NBMs (list/dict): repr_NBM[i] contains a dict with the pretrained NBM for WT i including threshold and statistics
        ignore_scarce_destination (bool): Set to True if no performance w.r.t mapping TO destination WT should be included (default set to True) 
    '''
    N_WTS = len(repr_NBMs) 
    results = {}
    for origin_wt in range(0, N_WTS):
        results[origin_wt] = {}
        performances = calculate_score_similarity(all_mapped_anomaly_scores[origin_wt], origin_wt, ground_truths, repr_NBMs, ignore_scarce_destination)
        results[origin_wt] = (performances)


    # calculate the ensemble fusion strategy performances for the target->sources directions
    ensemble_performance = calculate_ensemble_fusion(all_mapped_anomaly_scores[N_WTS-1], N_WTS-1, ground_truths, repr_NBMs)



    # calculate key metrics for src-to-src and target-to-src
    # source-to-source:
    src_to_src_summary = {}
    src_to_src_F1s = np.asarray([[d["F1"] for d in results[k]] for k in range(N_WTS-1)])
    src_to_src_summary["avg_F1"] = np.mean(src_to_src_F1s.flatten())
    src_to_src_summary["median_F1"] = np.median(src_to_src_F1s.flatten())
    src_to_src_summary["std_F1"] = np.std(src_to_src_F1s.flatten())
    src_to_src_summary["worst_F1"] = np.min(src_to_src_F1s.flatten())
    src_to_src_summary["best_F1"] = np.max(src_to_src_F1s.flatten())
    src_to_src_summary["all_F1s"] = src_to_src_F1s

    # scarce target -> source WTs
    scarce_summary = {}
    scarce_to_src_f1s = np.asarray([d["F1"] for d in results[N_WTS-1]]).flatten()
    scarce_summary["avg_F1"] = np.mean(scarce_to_src_f1s)
    scarce_summary["median_F1"] = np.median(scarce_to_src_f1s)
    scarce_summary["std_F1"] = np.std(scarce_to_src_f1s)
    scarce_summary["worst_F1"] = np.min(scarce_to_src_f1s)
    scarce_summary["best_F1"] = np.max(scarce_to_src_f1s)
    scarce_summary["all_F1s"] = scarce_to_src_f1s
    scarce_summary["ensemble_F1"] = ensemble_performance

    results["src_to_src_summary"] = src_to_src_summary
    results["scarce_summary"] = scarce_summary

    return results

def calculate_score_similarity(mapped_anomaly_scores_from_origin, origin_id, repr_truth, repr_NBMs, ignore_scarce_destination):
    '''
    Given mapped anomaly scores to all destinations from the same origin WT (origin id), calculates the 
    similarity performance (esp. F1) compared to the ground truth (repr_truth[origin_id].
    The anomaly scores are binarized and set in relation to the destination NBM's TH.

    Returns a list of performance metrics for every origin->destination direction.

    Args:
        mapped_anomaly_scores_from_origin (dict): 
        origin_id (int): id of the origin wt i
        repr_truth (list/dict): repr_truth[i] represents the ground truth / representative-trained anomaly scores of origin WT i 
        repr_NBMs (list/dict): repr_NBMs[j] contains the pretrained NBM and threshold of destination WT j
        ignore_scarce_destination (bool): If True: does not calculate the performance for mapping TO the scarce target domain (last WT in setups)

    '''

    N_WTS = len(repr_NBMs)
    from_repr_scores = repr_truth[origin_id]["scores"]
    from_repr_y = from_repr_scores >= repr_NBMs[origin_id]["TH"]

    performances = []
    for destination_id in [x for x in list(range(0, N_WTS)) if x != origin_id]:
        if ignore_scarce_destination and (destination_id == N_WTS -1): continue
        from_to_scores = mapped_anomaly_scores_from_origin[destination_id]["mapped_scores"]
        from_to_y = from_to_scores >= repr_NBMs[destination_id]["TH"] # binarized w.r.t. threshold
        performances.append(threshold_similarity_performance(from_repr_y, from_to_y))

    return performances


def calculate_ensemble_fusion(mapped_scarce_to_sources, origin_id, repr_truth, repr_NBMs):
    '''
    Given the mapped anomaly scores from the scarce to all sources, calculate the ensemble performance.
    Similarity performances (esp. F1) are calculated based on an ensemble of binarized anomaly scores.

    Minority vote, Majority voting, and Consensus are evaluated.

    Returns a dict of performance metrics for every ensemble fusion strategy.

    Args:
        mapped_scarce_to_sources (dict): Mapped anomaly scores all from target->source directions. 
        origin_id (int): The id of the scarce target WT.
        repr_truth (list/dict): repr_truth[i] represents the ground truth / representative-trained anomaly scores of origin WT i 
        repr_NBMs (list/dict): repr_NBMs[j] contains the pretrained NBM and threshold of destination WT j
    '''

    N_source_WTs = len(repr_NBMs) - 1
    from_repr_scores = repr_truth[origin_id]["scores"]
    # ground truth anomaly scores of the scarce target WT
    from_repr_y = from_repr_scores >= repr_NBMs[origin_id]["TH"]

    # collect all binarized target->source anomaly scores
    binarized_scores = []
    for destination_id in [x for x in list(range(0, N_source_WTs))]:
        from_to_scores = mapped_scarce_to_sources[destination_id]["mapped_scores"]
        from_to_y = (from_to_scores >= repr_NBMs[destination_id]["TH"]).astype(int) # binarized w.r.t. threshold

        binarized_scores.append(from_to_y)

    binarized_scores = np.asarray(binarized_scores)
    
    # calculate performance for ensemble fusion strategies
    ensemble_performances = {}

    # consensus
    consensus = np.all(binarized_scores == 1, axis=0)
    ensemble_performances["consensus"] = threshold_similarity_performance(from_repr_y, consensus)['F1']

    # majority voting (at least half to count as an anomaly)
    majority = np.sum(binarized_scores, axis=0)
    ensemble_performances["majority"] = threshold_similarity_performance(from_repr_y, (majority >= int(N_source_WTs/2)).astype(int))['F1']


    # minority strategy
    minority_result = np.any(binarized_scores == 1, axis=0).astype(int)
    ensemble_performances['minority'] = threshold_similarity_performance(from_repr_y, minority_result)['F1']

    return ensemble_performances