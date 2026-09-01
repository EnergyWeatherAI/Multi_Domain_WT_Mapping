# Multi_Domain_WT_Mapping
Code repository of our research article *"Generative Multi-Domain Transfer Learning for Fault Detection in Data-Scarce Wind Turbines"*, [available on arxiv](https://arxiv.org/abs/2608.30323).


### Data and structure:
We provide a *dummy* dataset in /dataset/ for 3 different farms, each with a separate SCADA dataset for its WT (1 random data row for illustrating our formatting, features, and structure). The sites are further described in the META.csv file with their rated power and rated wind speed. These single row datasets are not suitable to run the scripts, as they are only meant to illustrate the format and structure. Data that should be replaced with proprietary datasets before running the code. 


### Usage examples:

***Train NBMs***

Separately per WT. Representative (full training data) NBMs are required for every turbine, scarce NBMs only for the data-scarce NBM baseline.

    python train_NBM.py -SITE_NAME=farm1 -WT_ID=1 -CUDA_IDX=0
    python train_NBM.py -SITE_NAME=farm1 -WT_ID=1 -SCARCITY=2w -CUDA_IDX=0

`-SCARCITY` can take values in `{1w, 2w, 3w, 1m, 6w, 2m}`. Leave empty for full training data.


***Evaluate NBMs***

If both a representative NBM and data-scarce NBM were trained, the models and needed information are stored in `saves/`.
The following script evaluates the data-scarce NBM performance.

    python evaluate_scarce_NBM.py -SITE_NAME=farm1 -WT_ID=1 -SCARCITY=2w

***Train the StarGAN-based domain mapping model***

Setups (assigning which WTs act as source domains, and which one as target WT) are defined in `utils/setup_utils.py`.

    python train_mapping.py -SETUP_ID=1 -SCARCITY=2w -CUDA_IDX=0

***Evaluate the mapping***

Evaluates every trained multi-domain WT mapping model configuration found for the specified setup.

    python evaluate_mapping.py -SETUP_ID=1 -SCARCITY=2w -CUDA_IDX=0

Results are stored in `results/`
