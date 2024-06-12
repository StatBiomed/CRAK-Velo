import scvelo as scv
import scanpy as sc
#import pandas as pd

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
    adata = scv.read(config['adata_path'])
    #adata_atac = scv.read(config['atac_adata_path'])
    #df_rg_intersection = pd.read_csv(config['regions_genes_intersections'])

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

    return adata#, adata_atac, df_rg_intersection