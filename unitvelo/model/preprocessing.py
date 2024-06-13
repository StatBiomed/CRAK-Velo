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

    return adata#, adata_atac

# def genes_regions_interesctions(adata, adata_atac, config, col = ['chrom', 'chromStart', 'chromEnd'], w = None,):
    #ngenes = adata.X.shape[1]
    #nregions = adata_atac.X.shape[1]

    #if w == None:
           #w = config['preprocessing']['window']
    
    #gene_coor = adata.var.sort_values(col[0])[[col[0],col[1],col[2]]]
    #gene_coor["gene_name"] = gene_coor.index
    #gene_coor ["gene_number"] = np.arange(0,ngenes)

    #region_coor = adata_atac.var.sort_values(col[0])[[col[0],col[1],col[2]]]
    #region_coor ["region_number"] = np.arange(0,nregions)

    #a = pybedtools.BedTool.from_dataframe(region_coor)
    #b = pybedtools.BedTool.from_dataframe(gene_coor)
    #df_rg_intersection = a.window(b, w=w).overlap(cols=[2,3,6,7])

    #col_names = ["chrom_region", "start_region", "end_region", "region_number", "chrom_gene", "start_gene", "end_gene", "gene_name", "gene_number"," "]
    #df_rg_intersection.to_dataframe(names=col_names).iloc[:,:-1]

    #return df_rg_intersection


    #def gene_regions_binary_matrix(adata, adata_atac, df_rg_intersection):

        #ngenes = adata.X.shape[1]
        #nregions = adata_atac.X.shape[1]
        
        # columns = df_rg_intersection["gene_number"]
        # rows = df_rg_intersection["region_number"]
        # B = np.zeros((nregions, ngenes), dtype=int)
        # B[rows, columns] = 1
        # non_zero_regions = B.sum(axis=1) != 0
        # adata_atac = adata_atac[:, non_zero_regions]
        # B = B[non_zero_regions, :]
        # return B, adata_atac