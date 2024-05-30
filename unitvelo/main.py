from .velocity import Velocity
import scvelo as scv
import os

def run_model(
    adata,
    label,
    config_file=None,
    normalize=True,
):
    """Preparation and pre-processing function of RNA velocity calculation.
    
    Args:
        adata (str): 
            takes relative of absolute path of Anndata object as input or directly adata object as well.
        label (str): 
            column name in adata.var indicating cell clusters.
        config_file (object): 
            model configuration object, default: None.
        
    Returns:
        adata: 
            modified Anndata object.
    
    """

    from .utils import init_config_summary, init_adata_and_logs
    config, _ = init_config_summary(config=config_file)
    adata, data_path = init_adata_and_logs(adata, config, normalize=normalize)

    scv.settings.presenter_view = True
    scv.settings.verbosity = 0
    scv.settings.file_format_figs = 'png'

    adata.uns['datapath'] = data_path
    adata.uns['label'] = label
    adata.uns['basis'] = config.BASIS

    model = Velocity(adata, config=config)
    model.get_velo_genes()

    adata = model.fit_velo_genes()

    scv.tl.velocity_graph(adata, sqrt_transform=True)
    scv.tl.velocity_embedding(adata, basis=adata.uns['basis'])
    scv.tl.latent_time(adata, min_likelihood=None)

    adata.write(os.path.join(f'adata_fitted_{config.FIT_OPTION}.h5ad'))

    return adata