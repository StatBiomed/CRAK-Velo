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

## Running Crak-Velo
# Configurations
To run Crak-Velo you need to add the path of the .json files. Our .json files can be found in the config folder: 
./crak-velo/config/

You can replicate the .json files to use for other datasets. In the json files it is important to specify the path of the annotation rna and atac data, bed file produced by pybedtools, and the name of the key in obs that has the information about cell clusters.
Example from the header of a config.json file:
```bash
 "name": "VelocityDemo",
 "logger_config_path": "./CRAK-Velo/crak-velo/config/config_logger.json",

 "adata_path": "./CRAK-Velo/notebooks/data/HSPC_dataset/postpro_adata/adata_rna_postpro.h5ad",
 "adata_atac_path": "./CRAK-Velo/notebooks/data/HSPC_dataset/postpro_adata/adata_atac_postpro.h5ad",
 "df_rg_intersection_path": "./CRAK-Velo/notebooks/data/HSPC_dataset/df_rg_intersections.csv",
 "cluster_name": "celltype"
```
Also you need to specify the path for the saving directory and number of epochs (default 10e4):
```bash
 "base_trainer": {
        "epochs": 10000,
        "save_dir": "./CRAK-Velo_fit/", ....}
```
# Smoothing scATAC-seq data:
we used our implementation of cisTopic that you can find here:
 
# Running command
The path to the json file is needed and the window used to create the bed file (deafualt window length is 10e4).
Example:
```bash
 python ./crak-velo/main.py --config ./crak-velo/config/config_main_10X_mouse_brain.json --w 10000
```

## Tutorials notebooks
 Two tutorials notebooks is provided:

1- preprocessing.ipynb: shows preprocessing of the HSPC data and the smoothing of the scATAC-seq data using our cisTopic implementation.

2- producing_df_rg_intersection.ipynb: shows how to produce the df_rg_intersection.csv using pybedtools necessary for running CRAK-Velo.


## Figures notebooks
For each figure in the paper a notebook is provide:

Fig1.ipynb: for the HSPC dataset  
Fig1.ipynb: for Fresh Embryonic E18 Mouse Brain (5k) dataset



