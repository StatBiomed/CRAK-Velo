from velocity import Velocity
from model import init_adata, init_config, gene_regions_binary_matrix#, genes_regions_interesctions
import scvelo as scv
import os
from utils import ConfigParser, set_seed
import argparse
#import scanpy as sc
#import numpy as np
def run_model(config):
    set_seed(config['system']['seed'])
    # writer = SummaryWriter(log_dir=config._log_dir)

    logger = config.get_logger('trainer')
    logger.info('Inferring RNA velocity with scATAC-seq data')
    
    config = init_config(config=config)
    adata , adata_atac, df_rg_intersection = init_adata(config, logger, normalize=True) 
    
    #df_rg_intersection = genes_regions_interesctions(adata, adata_atac, config)
    #df_rg_intersection.to_csv('/data/nelkazwi/RNA_velo/Unitvelo_atac/df_regions_genes/10X_mouse_brain_100000.csv',sep='\t')
    B, adata_atac = gene_regions_binary_matrix(config, adata, adata_atac, df_rg_intersection, logger)
    logger.info(f"adata shape: {adata.shape} adata_atac shape: {adata_atac.shape} cisTopic shape: {adata_atac.obsm['cisTopic'].shape} binary matrix shape: {B.shape}\n")
    
    #np.savetxt('/data/nelkazwi/RNA_velo/Unitvelo_atac/test_multivelo_data/B.csv', B, delimiter='\t')
    scv.settings.presenter_view = True
    scv.settings.verbosity = 0
    scv.settings.file_format_figs = 'png'

    model = Velocity(adata, adata_atac, B, logger, config=config) 
    model.get_velo_genes()
    
    adata = model.fit_velo_genes()
    adata.uns['basis'] = config['preprocessing']['basis']
    
    # sc.pp.neighbors(adata)
    # sc.tl.umap(adata, n_components=2)
    # scv.tl.velocity_graph(adata, sqrt_transform=True)
    # scv.tl.velocity_embedding(adata, basis=config['preprocessing']['basis'])
    # scv.tl.latent_time(adata, min_likelihood=None)

    df = adata_atac.var
    adata.write(os.path.join(config.save_dir, f'model_last.h5ad'))
    adata_atac.write(os.path.join(config.save_dir, f'model_last_atac.h5ad'))
    df.to_csv(os.path.join(config.save_dir, f'regions_information.csv'), sep='\t')

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Velocity Estimation of scRNA-seq and scATAC-seq')
    args.add_argument('-c', '--config', default='config.json', type=str, help='config file path (default: None)')
    args.add_argument('-id', '--run_id', default=None, type=str, help='id of experiment (default: current time)')
    args.add_argument('-w', '--window', default= None, type=int, help='window around gene (default: None)')
    args.add_argument('-l', '--loss_mode', default= None, type=int, help='mode_of_loss_function: 1 or 2 (default: None)')
    config = ConfigParser.from_args(args)
    run_model(config)