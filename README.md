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
To run Crak-Velo you need to add the path of the .json files. Our .json files can be found in the config folder: 
./crak-velo/config/

You can replicate the .json files to use for other datasets. In the json files it is important to specify the path of the annotation rna and atac data and the bed files produced by pybedtools.
Example:
```bash
 python ./crak-velo/main.py --config ./crak-velo/config/config_main_10X_mouse_brain.json --w 10000
```


```bash
 python ./crak-velo/main.py --config ./crak-velo/config/config_main_10X_mouse_brain.json --w 10000
```




