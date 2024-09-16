# Crak-Velo

CRAK-Velo is a semi-mechanistic model that incorporates chromatin accessibility data into the estimation of RNA velocities. It offers biologically consistent estimates of developmental trajectories and enables precise cell-type deconvolution and examine the interactions between genes and chromatin regions.

## Installation
We run CRAK-Velo on the same environment we used to run UniTVelo installed using:

```bash
conda create -n unitvelo python=3.7
conda activate unitvelo
```

UniTVelo package can be conveniently installed via PyPI:

```bash
pip install unitvelo

```

We need to install pybedtools to intersect the regions in scATAC-seq data with genes in scRNA-seq data:

```bash
conda install --channel conda-forge --channel bioconda pybedtools
```

# Running Crak-Velo
## Configurations
To run Crak-Velo you need to set up .json files.
 Our .json files can be found in the config folder: 
./crak-velo/config/

You can replicate the .json files in the repo for usage in other datasets. In the json files it is important to specify the path of the annotation rna and atac data, and the name of the key in obs that has the information about cell clusters.
Example from the first rows of a config.json file:
```bash



"name": "VelocityDemo",
"logger_config_path": "./CRAK-Velo/crak-velo/config/config_logger.json",

"adata_path": "./CRAK-Velo/notebooks/data/HSPC_dataset/prepro_adata/adata_rna_prepro.h5ad",
"adata_atac_path": "./CRAK-Velo/notebooks/data/HSPC_dataset/postpro_atac_adata/adata_atac_postpro.h5ad",

"cluster_name": "celltype"}
```
Also the path for the saving directory and number of epochs (default 10e4) need to be specified:
```bash
 "base_trainer": {
        "epochs": 10000,
        "save_dir": "./CRAK-Velo_fit/", ....}
```
## Smoothing scATAC-seq data:
we used our implementation of cisTopic that you can download:
```bash
git clone https://github.com/Nour899/cisTopic.git
```
 Note that we used different environmet to run cisTopic as it requires torch. The environment.yml is shown in cisTopic repo.


## Running command
The path to the json file is needed and the window used to create the bed file (deafualt window length is 10e4).
Example:
```bash
 python ../../crak-velo/main.py --config ../../crak-velo/config/config_main_HSPC.json --w 10000
```

# Preprocessing of scATAC-seq data
 A notebook for preprocessing is provided. It shows the preprocessing and smoothing of the of the scATAC-seq data in HSPC data. The smoothing is done using our cisTopic implementation. Also, it shows the required format of .var of the input anndata to get the genes-regions intersections.

# Figures notebooks
For each figure in the paper a notebook is provide:

Fig1.ipynb: for the HSPC dataset  
Fig2.ipynb: for Fresh Embryonic E18 Mouse Brain (5k) dataset

# CRAK-Velo outputs
The code will produce three outputs:

adata_rna_fit.h5ad: contains information of the fitted parameters by CRAK-Velo in .var. For the weights they are stored in .varm["fit_region_weights"].

adata_atac_fit.h5ad: contains all the regions used in the optimization.

B.txt: binary matrix that has dimension regions by genes. It indicates which regions are used to optimize for the gene specific parameters.




