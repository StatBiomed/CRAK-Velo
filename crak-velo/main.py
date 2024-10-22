from velocity import Velocity
from model import init_adata, init_config, gene_regions_binary_matrix, genes_regions_interesctions
import os
from utils import ConfigParser, set_seed
import argparse
import numpy as np
def run_model(config):
    set_seed(config['system']['seed'])
    

    logger = config.get_logger('trainer')
    logger.info('Inferring RNA velocity with scATAC-seq data')
    
    config = init_config(config=config)
    adata , adata_atac = init_adata(config, logger, normalize=True) 
    
    df_rg_intersection = genes_regions_interesctions(adata,adata_atac, config)
    print(df_rg_intersection.shape)
    B, adata_atac = gene_regions_binary_matrix(config, adata, adata_atac, df_rg_intersection, logger)
    
    logger.info(f"adata shape: {adata.shape} adata_atac shape: {adata_atac.shape} cisTopic shape: {adata_atac.obsm['cisTopic'].shape} binary matrix shape: {B.shape}\n")
    
    


    model = Velocity(adata, adata_atac, B, logger, config=config) 
    model.get_velo_genes()
    
    adata = model.fit_velo_genes()
    adata.uns['basis'] = config['preprocessing']['basis']
    
    

    
    adata.write(os.path.join(config.save_dir, f'adata_rna_fit.h5ad'))
    adata_atac.write(os.path.join(config.save_dir, f'adata_atac_fit.h5ad'))
    np.savetxt(os.path.join(config.save_dir, f'B.txt'), B, delimiter=',')

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Velocity Estimation of scRNA-seq and scATAC-seq')
    args.add_argument('-c', '--config', default='config.json', type=str, help='config file path (default: None)')
    args.add_argument('-id', '--run_id', default=None, type=str, help='id of experiment (default: current time)')
    args.add_argument('-w', '--window', default= None, type=int, help='window around gene (default: None)')
    
    config = ConfigParser.from_args(args)
    run_model(config)