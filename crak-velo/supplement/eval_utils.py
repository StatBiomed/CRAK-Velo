"""
Evaluation utility functions.
This module contains util functions for computing evaluation scores.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE



def summary_scores(all_scores):
    """Summarize group scores.
    
    Args:
        all_scores (dict{str,list}): 
            {group name: score list of individual cells}.
    
    Returns:
        dict{str,float}: 
            Group-wise aggregation scores.
        float: 
            score aggregated on all samples
        
    """
    sep_scores = {k:np.mean(s) for k, s in all_scores.items() if s}
    overal_agg = np.mean([s for k, s in sep_scores.items() if s])
    return sep_scores, overal_agg

def keep_type(adata, nodes, target, k_cluster):
    """Select cells of targeted type
    
    Args:
        adata (Anndata): 
            Anndata object.
        nodes (list): 
            Indexes for cells
        target (str): 
            Cluster name.
        k_cluster (str): 
            Cluster key in adata.obs dataframe

    Returns:
        list: 
            Selected cells.

    """
          
    return nodes[adata.obs[k_cluster][nodes].values == target]

def cross_boundary_correctness(
    adata, 
    k_cluster, 
    k_velocity, 
    cluster_edges, 
    return_raw=False, 
    x_emb="X_umap"
):
    """Cross-Boundary Direction Correctness Score (A->B)
    
    Args:
        adata (Anndata): 
            Anndata object.
        k_cluster (str): 
            key to the cluster column in adata.obs DataFrame.
        k_velocity (str): 
            key to the velocity matrix in adata.obsm.
        cluster_edges (list of tuples("A", "B")): 
            pairs of clusters has transition direction A->B
        return_raw (bool): 
            return aggregated or raw scores.
        x_emb (str): 
            key to x embedding for visualization.
        
    Returns:
        dict: 
            all_scores indexed by cluster_edges or mean scores indexed by cluster_edges
        float: 
            averaged score over all cells.
        
    """
    scores = {}
    all_scores = {}
    
    if x_emb == "X_umap":
        v_emb = adata.obsm['{}_umap'.format(k_velocity)]
    else:
        v_emb = adata.obsm[[key for key in adata.obsm if key.startswith(k_velocity)][0]]

    x_emb = adata.obsm[x_emb]
        
    for u, v in cluster_edges:
        sel = adata.obs[k_cluster] == u
        nbs = adata.uns['neighbors']['indices'][sel] # [n * 30]
       
        d = np.unique(adata.uns['neighbors']['indices'][sel].flatten())
    
        
        boundary_nodes = map(lambda nodes:keep_type(adata, nodes, v, k_cluster), nbs)
     
        x_points = x_emb[sel]
        x_velocities = v_emb[sel]

        type_score = []
        uni_nei = adata[d,:][adata[d,:].obs[k_cluster] == v]
        d = uni_nei.shape[0]
        b = adata[adata.obs[k_cluster] == v].shape[0]
       
        for x_pos, x_vel, nodes in zip(x_points, x_velocities, boundary_nodes):
            if len(nodes) == 0: continue
           
            position_dif = x_emb[nodes] - x_pos
            dir_scores = cosine_similarity(position_dif, x_vel.reshape(1,-1)).flatten()
            type_score.append(np.mean(dir_scores))
        
        scores[(u, v)] = np.mean(type_score)
        all_scores[(u, v)] = type_score
        
       
       
    if return_raw:
        return all_scores 
    
    return scores, np.mean([sc for sc in scores.values()])

def inner_cluster_coh(adata, k_cluster, k_velocity, return_raw=False):
    """In-cluster Coherence Score.
    
    Args:
        adata (Anndata): 
            Anndata object.
        k_cluster (str): 
            key to the cluster column in adata.obs DataFrame.
        k_velocity (str): 
            key to the velocity matrix in adata.obsm.
        return_raw (bool): 
            return aggregated or raw scores.
        
    Returns:
        dict: 
            all_scores indexed by cluster_edges mean scores indexed by cluster_edges
        float: 
            averaged score over all cells.
        
    """
    clusters = np.unique(adata.obs[k_cluster])
    scores = {}
    all_scores = {}

    for cat in clusters:
        sel = adata.obs[k_cluster] == cat
        nbs = adata.uns['neighbors']['indices'][sel]
        same_cat_nodes = map(lambda nodes:keep_type(adata, nodes, cat, k_cluster), nbs)

        velocities = adata.layers[k_velocity]
        cat_vels = velocities[sel]
        cat_score = [cosine_similarity(cat_vels[[ith]], velocities[nodes]).mean() 
                     for ith, nodes in enumerate(same_cat_nodes) 
                     if len(nodes) > 0]
        all_scores[cat] = cat_score
        scores[cat] = np.mean(cat_score)
    
    if return_raw:
        return all_scores
    
    return scores, np.mean([sc for sc in scores.values()])

def evaluate(
    adata, 
    cluster_edges, 
    k_cluster, 
    k_velocity="velocity", 
    x_emb="X_umap", 
    verbose=True
):
    """Evaluate velocity estimation results using 5 metrics.
    
    Args:
        adata (Anndata): 
            Anndata object.
        cluster_edges (list of tuples("A", "B")): 
            pairs of clusters has transition direction A->B
        k_cluster (str): 
            key to the cluster column in adata.obs DataFrame.
        k_velocity (str): 
            key to the velocity matrix in adata.obsm.
        x_emb (str): 
            key to x embedding for visualization.
        
    Returns:
        dict: 
            aggregated metric scores.
    
    """

    from .eval_utils import cross_boundary_correctness
    from .eval_utils import inner_cluster_coh
    crs_bdr_crc = cross_boundary_correctness(adata, k_cluster, k_velocity, cluster_edges, True, x_emb)
    ic_coh = inner_cluster_coh(adata, k_cluster, k_velocity, True)
    
    if verbose:
        print("# Cross-Boundary Direction Correctness (A->B)\n{}\nTotal Mean: {}".format(*summary_scores(crs_bdr_crc)))
        print("# In-cluster Coherence\n{}\nTotal Mean: {}".format(*summary_scores(ic_coh)))
    
    return {
        "Cross-Boundary Direction Correctness (A->B)": crs_bdr_crc,
        "In-cluster Coherence": ic_coh,
    }
# %%


def class_cu (adata, gene_list, test_size, key):  
   G = len(gene_list)
   acc = np.zeros(G)
   
   for w, gene_name in enumerate(gene_list):
   
    adata.var[adata.var_names == gene_name]#['velocity_genes']
    gene_number = np.where(adata.var_names == gene_name)[0][0]

    u = adata.layers['unspliced'][:,gene_number].toarray()
    c = adata.layers['ATAC'][:,gene_number][:,np.newaxis]
    cu = np.concatenate((c, u), axis=1)

    X_train, X_test, y_train, y_test = train_test_split(cu, adata.obs[key].values, test_size = test_size)
    classifier = KNeighborsClassifier( metric = 'euclidean')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    
    ac = accuracy_score(y_test,y_pred)
    acc[w]=ac

   return acc 


def class_su (adata, gene_list, test_size, key):  
   G = len(gene_list)
   acc = np.zeros(G)
  
   for w, gene_name in enumerate(gene_list):

    adata.var[adata.var_names == gene_name]#['velocity_genes']
    gene_number = np.where(adata.var_names == gene_name)[0][0]

    u = adata.layers['unspliced'][:,gene_number].toarray()
    s = adata.layers['spliced'][:,gene_number].toarray()
    su = np.concatenate((s, u), axis=1)

    X_train, X_test, y_train, y_test = train_test_split(su, adata.obs[key].values, test_size = test_size)
    classifier = KNeighborsClassifier( metric = 'euclidean')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    
    
    ac = accuracy_score(y_test,y_pred)
    acc[w]=ac
   return acc 


def KNN_eval (adata, adata_mv, key1, key2, test_size = 0.30):

    u_dot_atac = adata.layers['u_atac'] ##remove genes that has no regions around it
    adata = adata[:,u_dot_atac.sum(axis=0) != 0]
    gene_list = np.intersect1d(adata.var_names, adata_mv.var_names)

    acc = class_cu (adata, gene_list, test_size, key1)
    acc_mv = class_cu (adata_mv, gene_list, test_size,key2)
    acc_su = class_su (adata, gene_list, test_size,  key1)

    return acc, acc_mv, acc_su
