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

    if config.BASIS is None:
        basis_keys = ["pca", "tsne", "umap"]
        basis = [key for key in basis_keys if f"X_{key}" in adata.obsm.keys()][-1]
    elif f"X_{config.BASIS}" in adata.obsm.keys():
        basis = config.BASIS
    else:
        raise ValueError('Invalid embedding parameter config.BASIS')
    adata.uns['basis'] = basis

    model = Velocity(adata, config=config)
    model.get_velo_genes()

    adata = model.fit_velo_genes(basis=basis)
    adata.write(os.path.join(adata.uns['temp'], f'temp_{config.FIT_OPTION}.h5ad'))
    
    if 'examine_genes' in adata.uns.keys():
        from .individual_gene import exam_genes
        exam_genes(adata, adata.uns['examine_genes'])

    return adata