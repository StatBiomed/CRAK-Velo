from velocity import Velocity
from model import init_adata, init_config, gene_regions_binary_matrix
import scvelo as scv
import os
from utils import ConfigParser, set_seed
import argparse

def run_model(config):
    set_seed(config['system']['seed'])
    # writer = SummaryWriter(log_dir=config._log_dir)

    logger = config.get_logger('trainer')
    logger.info('Inferring RNA velocity with scATAC-seq data')
    
    config = init_config(config=config)
    adata , adata_atac, df_rg_intersection = init_adata(config, logger, normalize=True) 
    
    #df_rg_intersection = genes_regions_interesctions(adata, adata_atac, config)
    B, adata_atac = gene_regions_binary_matrix(adata, adata_atac, df_rg_intersection, logger)
    logger.info(f"adata shape: {adata.shape} adata_atac shape: {adata_atac.shape} cisTopic shape: {adata_atac.obsm['cisTopic'].shape} binary matrix shape: {B.shape}\n")
      
    scv.settings.presenter_view = True
    scv.settings.verbosity = 0
    scv.settings.file_format_figs = 'png'

    model = Velocity(adata, adata_atac, B, logger, config=config) 
    model.get_velo_genes()
    
    adata = model.fit_velo_genes()
    # adata.uns['basis'] = config['preprocessing']['basis']

    # scv.tl.velocity_graph(adata, sqrt_transform=True)
    # scv.tl.velocity_embedding(adata, basis=config['preprocessing']['basis'])
    # scv.tl.latent_time(adata, min_likelihood=None)

    adata.write(os.path.join(config.save_dir, f'model_last.h5ad'))

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Velocity Estimation of scRNA-seq and scATAC-seq')
    args.add_argument('-c', '--config', default='config.json', type=str, help='config file path (default: None)')
    args.add_argument('-id', '--run_id', default=None, type=str, help='id of experiment (default: current time)')

    config = ConfigParser.from_args(args)
    run_model(config)