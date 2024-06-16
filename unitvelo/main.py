from velocity import Velocity
from model import init_adata, init_config, genes_regions_interesctions, gene_regions_binary_matrix
import scvelo as scv
import os
from utils import ConfigParser, set_seed
import argparse
import logging

def run_model(config):
    set_seed(config['system']['seed'])
    # writer = SummaryWriter(log_dir=config._log_dir)

    logger = logging.getLogger('trainer')
    logger.info('Inferring RNA velocity with scATAC-seq data')
    logger.info(f"Using adata file from {config['adata_path']}\n")
    
    config = init_config(config=config)
    adata , adata_atac = init_adata(config, logger, normalize=True) ##
    df_rg_intersection = genes_regions_interesctions(adata, adata_atac, config)
    B, adata_atac = gene_regions_binary_matrix(adata, adata_atac, df_rg_intersection)
      
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

    # adata.write(os.path.join(config.save_dir, f'model_last.h5ad'))

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Velocity Estimation of scRNA-seq and scATAC-seq')
    args.add_argument('-c', '--config', default='config.json', type=str, help='config file path (default: None)')
    args.add_argument('-id', '--run_id', default=None, type=str, help='id of experiment (default: current time)')

    config = ConfigParser.from_args(args)
    run_model(config)