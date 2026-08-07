import torch 
import random

###
# Used to perform the anomaly augmentation (see Appendix of paper):
# At training time, a batch is copied and subsequently artificially corrupted. 
# The random corruption process sets a random part of random channels to its normalization minimum (e.g., 0 power) to break correlations 
# to encourage the domain mapping network to generate non-normal data as well.
###

def corrupt_batch(x):
    '''
    Copies a torch batch x (to avoid modifying inplace) and returns an artificially corrupted batch.

    points_to_corrupt determines the length of the corruption segment starting at starting_point
    for the randomly selected ch_to_corrupt group.
    '''
    transf = x.clone()
    with torch.no_grad():
        # choose channel group to corrupt (either entire power, speed, or rotor group)
        ch_to_corrupt = random.randint(0, 2)
        # choose proportion (length-wise) to corrupt, minimally 60%, maximally 100% (entire channel)
        points_to_corrupt = int(72 * (random.randint(60, 100) * 0.01))
        starting_point = random.randint(0, 72 - points_to_corrupt)

        # perform the transformation over the entire batch and return the batch
        transf = zero_out_segment(transf, ch_to_corrupt, points_to_corrupt, starting_point)

    return transf


def zero_out_segment(x, channel, pts_to_crr, start):
    '''
    Performs the specified corruption over channel(s) of a given batch.

    Args:
        x (torch tensor): The batch of data
        channel (int): The randomly selected channel to corrupt
        pts_to_crr (int): How many points to corrupt (in length, i.e., up to 72)
        start (int): At which point to start the corruption (i.e., up to 72-pts_to_crr)

    The selected channel determines whether it belongs to the power, windspeed, rotor group 
    (which may corrupt up to 3 channels).
    NOTE: assumption here is that the sample features are in order of: 
    [power_min, power_avg, power_max, ws_min, ws_avg, ws_max, rtr_min, rtr_avg, rtr_max, temp1, temp2]
    
    The manually adjusted normalization process (see appendix of paper) ensures that minimum of the power, ws, and rotor speed is 0.
    Therefore, this process will set these groups to 0 power, windspeed, or rotor speed. (-1.0 in normalized value)

    No corruption for the temperature.
    '''

    extra = random.randint(1, 3) # randomly set how many channels to corrupt within that group

    x[..., channel*3:(channel*3) + extra, start:start+pts_to_crr] = -1.0
    return x