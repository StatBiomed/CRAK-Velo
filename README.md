# Crak-Velo

CRAK-Velo is a semi-mechanistic model that incorporates chromatin accessibility data into the estimation of RNA velocities. It offers biologically consistent estimates of developmental trajectories and enables precise cell-type deconvolution and examine the interactions between genes and chromatin regions.

## Installation
We run CRAK-Velo on the same environment we used to run UniTVelo installed using:

```bash
conda create -n unitvelo python=3.7
conda activate unitvelo
```

UniTVelo package can be conveniently installed via PyPI:

```python3
pip install unitvelo

```

We need to install pybedtools to intersect the regions in scATAC-seq data with genes in scRNA-seq data:

```python3
conda install --channel conda-forge --channel bioconda pybedtools
```





