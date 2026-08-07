# helper functions for the setups (see paper)
# A setup consists of N source WTs and one data-scarce target WT (last WT)
def get_setup_sites_ids(setup_id):
    '''
    Returns the site names in a list and the corresponding WT_ids for each site as a list
    (for the purposes of our experimets, only the first WT in each farm is considered, i.e., WT ID 1)
    '''
    match setup_id:
        case 1:
            sites = ["farm4", "farm3", "farm2", "farm1"] # NOTE: The last WT represents the scarce target WT
            return sites, [1 for x in sites] # take the first WT (id 1) from each site
        case 2:
            sites = ["farm4", "farm3", "farm1", "farm2"]
            return sites, [1 for x in sites]
        case _:
            return None
        
