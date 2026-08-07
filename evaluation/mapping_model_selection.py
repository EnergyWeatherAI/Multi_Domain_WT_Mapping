import numpy as np

###############################
# Functions to calculate the presented model selection proxy metric.
# The mass (sum) of real anomaly scores and mapped anomaly scores are compared across all source-to-source WT directions.
###############################

def get_all_deltaT(mapped_anomaly_scores, ground_truths):
    '''
    For all possible source-to-source WT directions:
        Calculates the relative tail sum difference between real anomaly scores of a destination WT,
        and anomaly scores that have been mapped from an origin WT to that destination WT.
    
    Args:
        mapped_anomaly_scores (dict): 
            A dictionary containing anomaly scores obtained by mapping samples from an origin WT to a destination WT.
            The scores have been evaluated with the destination WT's representative NBM.
            mapped_anomaly_scores[0][1] are mapped anomaly scores obtained by mapping samples from origin WT to destination WT 1.  

        ground_truths (dict): ground_truths[i] represents the real anomaly scores evaluated by NBM i for destination WT i (ground truth)

    As in our paper, the expectation is to provide anomaly scores from mapping normal validation data across only data-rich source WTs.
    All scarce target WT's anomaly scores are ignored by default.
    '''

    N_WTS = len(ground_truths)
    delta_Ts = {}
 
    # calculate and collect the results for each source-to-source direction:
    for origin_wt_id in range(0, N_WTS-1): # ignoring the scarce WT 
        delta_Ts[origin_wt_id] = {}

        for destination_wt_id in range(0, N_WTS-1):
            if origin_wt_id == destination_wt_id: continue 

            # calculate delta T for the mapping quality of origin id *to* destination id
            mapped_scores_to_destination = mapped_anomaly_scores[origin_wt_id][destination_wt_id]["mapped_scores"]
            
            real_destination_scores = ground_truths[destination_wt_id]["scores"]
            
            deltaT = delta_T_difference_between_scores(real_destination_scores, mapped_scores_to_destination, 95)

            delta_Ts[origin_wt_id][destination_wt_id] = deltaT

    all_deltas = np.array([list(delta_Ts[k].values()) for k in delta_Ts]).flatten()

    # calculate key summaries for model selection
    delta_Ts["mean"] = np.mean(all_deltas)
    delta_Ts["median"] = np.median(all_deltas)
    delta_Ts["all_deltas"] = all_deltas
    return delta_Ts


def delta_T_difference_between_scores(real_scores, mapped_scores, tail_percentile = 95):
    '''
    The actual deltaT proxy metric calculation between two anomaly scores.
    Returns the relative difference of the sum of anomaly scores past the 95th percentile (set by real destination scores). 
    Args:
        real_scores (numpy array): The real anomaly scores of the destination WT
        mapped_scores (numpy array of same shape): Anomaly scores from samples mapped from origin->destination, evaluated by destination WT's NBM 
    '''
    th95 = np.percentile(real_scores, tail_percentile) 
    real_tail_mass = real_scores[real_scores >= th95].sum()
    mapped_tail_mass =  mapped_scores[mapped_scores >= th95].sum()

    return np.abs(real_tail_mass - mapped_tail_mass) / real_tail_mass