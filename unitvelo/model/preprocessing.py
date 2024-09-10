import scvelo as scv
import scanpy as sc
import pandas as pd
import numpy as np

def init_config(config=None):
    if config['fitting_option']['mode'] == 1:
        config['fitting_option']['density'] = "SVD" 
        config['fitting_option']['reorder_cell'] = 'Soft_Reorder'
        config['fitting_option']['aggregrate_t'] = True

    elif config['fitting_option']['mode'] == 2:
        config['fitting_option']['density'] = "Raw"
        config['fitting_option']['reorder_cell'] = 'Hard'
        config['fitting_option']['aggregrate_t'] = False

    else:
        raise ValueError('config.fitting_option.mode is invalid')
    
    return config

def init_adata(config, logger, normalize=True):
    logger.info(f"Using adata file from {config['adata_path']}")
    logger.info(f"Using adata_atac file from {config['adata_atac_path']}\n")

    adata = scv.read(config['adata_path'])
    adata_atac = scv.read(config['adata_atac_path'])
    df_rg_intersection = pd.read_csv(config['df_rg_intersection_path'], delimiter= "\t", index_col=0)

    if normalize:
        scv.pp.filter_and_normalize(
            adata, 
            min_shared_counts=config['preprocessing']['min_shared_counts'],
            n_top_genes=config['preprocessing']['n_top_genes']
        ) 

        logger.info(f"Extracted {adata.var[adata.var['highly_variable'] == True].shape[0]} highly variable genes")
        logger.info(f"Computing moments for {len(adata.var)} genes with n_neighbors: {config['preprocessing']['n_neighbors']} and n_pcs: {config['preprocessing']['n_pcs']}\n")

        sc.pp.pca(adata)
        sc.pp.neighbors(adata, n_pcs=config['preprocessing']['n_pcs'], n_neighbors=config['preprocessing']['n_neighbors'])
        scv.pp.moments(adata, n_pcs=None, n_neighbors=None)

    else:
        scv.pp.neighbors(adata)

    return adata, adata_atac, df_rg_intersection

def genes_regions_interesctions(
    df_rg_intersection, 
    config):

    w = config['preprocessing']['window']
    
    col_names = [
        "chrom_region", "start_region", "end_region", "region_number", 
        "chrom_gene", "start_gene", "end_gene", "gene_name", "gene_number", " "
    ]
    df_rg_intersection = df_rg_intersection.to_dataframe(names=col_names).iloc[:, :-1]
    
    df_rg_intersection["distance"] = np.abs(df_rg_intersection["start_gene"] - df_rg_intersection["start_region"])
    
    condition = df_rg_intersection["distance"] <= w
    

    df_rg_intersection = df_rg_intersection[condition]

    return df_rg_intersection

def gene_regions_binary_matrix(config, adata, adata_atac, df_rg_intersection, logger):
    ngenes = adata.X.shape[1]
    nregions = adata_atac.X.shape[1]
    logger.info(f"Creating binary matrix B of shape {nregions} x {ngenes}")
    
    B = np.zeros((nregions, ngenes), dtype=int)
    B[df_rg_intersection["region_number"], df_rg_intersection["gene_number"]] = 1

    non_zero_regions = B.sum(axis=1) == config['preprocessing']['reg_threshold'] 
    

    adata_atac.obsm["cisTopic"] = adata_atac.obsm["cisTopic"][:, non_zero_regions]
    
    return B[non_zero_regions, :], adata_atac[:, non_zero_regions]