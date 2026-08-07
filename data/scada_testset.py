import numpy as np 
from data.csv_loader import load_csv_as_df
import data.torch_data as torch_data
import torch

###
# An instance of a SCADA_Testset: Takes a configuration defining the processing 
# of SCADA data from *one WT* into *test/evaluation* data.
# Given a csv file, it is extracted into *uncleaned, unfiltered* test set SCADA samples,
# which may be potentially anomalous.
# Defined to be of certain shape (included channels and length). The data is processed from a csv
# into pandas dataframes, numpy arrays, and torch datasets and dataloaders prepared for evaluation.
#
# NOTE: Only test/evaluation data will be extracted by this class. See the SCADA_Trainingset documentation first.
###

class SCADA_Testset:
    '''
    Class for handling the data sequencing and processing, and for retrieving datasets of one specific WT according to a configuration.

    Attributes for initialization:
        config (dict): A dictionary containing configuration settings for extracting the data in a specific manner. 
        wt_label (torch): A torch-one-hot encoded label specifying this WT id, only used in StarGAN-based domain mapping
        scada_csv_path (str): The csv filepath corresponding to the WT's .csv file
        meta_csv_path (str): The csv filepath corresponding to a meta file, containing the rated wind speeds and power for all sites (see dataset example).


    This class handles the processing and retrieval of all forms of data for one WT.
    Its main method is get_data(), which returns a dictionary containing the extracted and processed data. Includes dataframes, normalized sequences, normalization functions, and torch data-sets/loaders. 
    Please refer to method documentation.
    '''

    def __init__(self, config, wt_label, scada_csv_path):
        '''
        Creates a dataset instance for the processing and retrieval of all forms of data for one WT.
        
        Args:
            config (dict): 
                    NOTE: It is expected to match the shared configuration values from the training set.
                    Expected config keys and values:
                        1) To specify the processing and sequencing of raw SCADA data:
                        x_features (list of str): A list of variable names to use as features within the sample, e.g. ["WindSpeed_min", "WindSpeed_avg", ...]
                        seq_len (int): The sequence length of a single SCADA sample. In our work, this is set to 72 timesteps (72 * 10 minutes = 12 hours) 
                        test_size (float): The test set size, as relative size of the entire loaded SCADA dataframe (0.30 for 30%)

                        tr_stats (dict or None): 
                            The data will be normalized according to provided statistics, which are expected to be the training set statistics from the domain mapping model.
                            For an NBM evaluation, the provided tr_stats should match the normalization statistics used by the NBM model.
            
                        2) Torch-specific configuration
                        bs (int): batch size for the torch test dataloader

            scada_csv_path (str): The path to the raw SCADA .csv file to process

        See the evaluation script (evaluate_mapping.py) for use examples.
        '''

        self.config = config 
        self.wt_label = wt_label # torch one hot encoded
        self.apply_filters = False # NOTE: No data filtering will be applied in this class, to retain possibly anomalous and unfiltered data. 
        self.scada_csv_path = scada_csv_path


    def get_data(self):
        '''
        Returns a dictionary containing processed SCADA data according to the class config attributes.
        '''
                
        # load dataframe, split into test df, extract sequences (e.g., 12h sequences)
        test_df, test_data = self.extract_sequences() 

        # obtain (de-)normalization function from the provided training statistics 
        norm_fX, unnorm_fX = self.get_normalization(self.config["tr_stats"])

        # convert into torch a dataset and dataloader, supply the (not yet applied) normalization function to torch
        test_ds = self.get_torch_datasets(test_data, self.wt_label, norm_fX)
        test_dl = self.get_torch_dataloaders(test_ds)

        # return results of all processing steps as dictionary
        scada_sample_dataset =  {
                "dfs": {"test": test_df},
                "data": {"test": test_data},
                "torch_datasets": {"test": test_ds},
                "torch_dataloaders": {"test": test_dl},
                "norm_stats": self.config["tr_stats"], "norm_fX": norm_fX, "unnorm_fX": unnorm_fX,
            }

        return scada_sample_dataset
    

    def extract_sequences(self):
        '''Returns a loaded test dataframe, and extracted sequences (e.g., 12h) obtained using a sliding window approach.'''
        
        # convert the csv file into a large dataframe
        wt_df = self.load_df()
        # split into the matching test set according to the configuration
        test_df = self.get_test_df(wt_df)

        # split into sequences, with corresponding timestamps of the last value and incident_flags for each sequence
        test_seq, test_last_timestamps, test_incident_flags  = self.split_into_sequences_extra(test_df)

        test_data = {"sequences": test_seq, "last_timestamps": test_last_timestamps, "incident_flags": test_incident_flags, "stats": self.config["tr_stats"]}
        return test_df, test_data


    def split_into_sequences_extra(self, df, convert_to_ch_first = True):
        '''
        Splits a provided dataframe into sequences (e.g., 12h-samples)

        Args:
            df (pandas dataframe): The dataframe from which to extract sequences
            convert_to_ch_first (bool): By default, this procedure extracts sequences in the shape of datapoints (length) x features (channels). If true, convert to Torch-preferred ch x l
        '''
        # convert dataframe into a list, from then on a sliding window approach
        df_dict = df.to_dict(orient="list")
        x_seq = []
        last_timestamps, incident_flags = [], []

        # 1) for each row value n, extract the [n+seq_len] range.
        # 2) Only retain the specified features in the config x_features list
        # 3) Only add the features to the x_seq list if there is no NA value within the sequence
        # 4) Additionally, extract and retain the timestamp of the last value and whether there was an incident within the sequence (anywhere within e.g., 12h) 
        for row in range(len(df_dict["Timestamp"])-(self.config["seq_len"]+1)):
            # x features
            features = {}
            for f in self.config["x_features"]: 
                features[f] = df_dict[f][row:row+self.config["seq_len"]]
            
            x = np.dstack(([features[f] for f in self.config["x_features"]])).reshape(self.config["seq_len"], -1)
                
            if np.isnan(x).sum() == 0:
                x_seq.append(x)
                last_timestamps.append(df_dict["Timestamp"][row:row+self.config["seq_len"]][-1])
                incident_flags.append(np.array(df_dict["incident"][row:row+self.config["seq_len"]]).max())
                
        if convert_to_ch_first: 
            return np.moveaxis(np.asarray(x_seq), 1, 2), last_timestamps, np.asarray(incident_flags)
        else: 
            return np.asarray(x_seq), last_timestamps, np.asarray(incident_flags)



    def load_df(self, keep_incident_flag=True):
        # converts the SCADA csv into a dataframe, with filters applied, depending on settings. See csv_loader.
        return load_csv_as_df(self.scada_csv_path, self.config["x_features"], None, None, self.apply_filters, keep_incident_flag) 


    def get_normalization(self, stats):
        '''
        Returns a torch-adapted transformation function to normalize (norm_fX) and de-normalize (unnorm_fX). To be provided for dataset instances.
        '''
        mins, maxs = torch.from_numpy(np.asarray(stats["X_mins"]).copy()).type(torch.float32), torch.from_numpy(np.asarray(stats["X_maxs"]).copy()).type(torch.float32)
        norm_fX = torch_data.get_normalize_f(mins, maxs, b=1, a=-1)
        unnorm_fX = torch_data.get_unnormalize_f(mins, maxs, b=1, a=-1)
        return norm_fX, unnorm_fX


    def get_test_df(self, wt_df, verbose=False):
        '''
        Splits a WT SCADA dataframe into a test dataframe (according to configuration) and returns it.
        '''

        # test size is INDEPENDENT of training (subset-) size. The last test_size*100% of data.
        # NOTE: Should ideally match the size specified to leave aside in the training sets.
        n_test_set = int(self.config["test_size"] *len(wt_df)) 
        test_df = wt_df[-n_test_set:]
        if verbose: print(f"Original DF lens: {len(test_df)}") # print out dataframe sizes for checks
        return test_df


    def get_torch_datasets(self, test_data, wt_label, transform):
        '''
        Converts the extracted sequence samples into a torch dataset (see torch_data.py).
        Args:
            test_data (numpy arrays): for test set sequences
            transform (torch transformation function): In our work, a normalization function adapated for torch tensors to normalizate an item from the dataset with.
        '''
        test_ds = torch_data.get_torch_datasets([test_data["sequences"]], wt_label = wt_label, transform = transform)
        return test_ds


    def get_torch_dataloaders(self, test_ds):
        '''
        Converts the torch dataset into a dataloader with specific batch sizes and shuffle properties.
        Args:
            test_ds (torch dataset): A torch dataset object , see get_torch_datasets and torch_data.py
            drop_last (bool): If set to true, the last batch is discarded. Should be set to false for full evaluation.
        '''
        test_dl = torch_data.get_dataloaders([test_ds], batch_sizes=[self.config["bs"]], shuffles=[False], drop_last = [False]) 
        return test_dl