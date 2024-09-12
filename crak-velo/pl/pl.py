import matplotlib.pyplot as plt
import scvelo as scv
import seaborn as sns
import numpy as np
import os
from scipy import sparse
import pandas as pd

def get_scatter_markers(n):
    markers = ['o', 's', 'D', '^', 'v', 'p', '*', '+', 'x', '|', '_']
    return markers[:n]  # Return a list of markers, truncating if necessary


def rbf(x, height, sigma, tau, offset_rbf):
    return height * np.exp(-sigma * (x - tau) * (x - tau)) + offset_rbf

def rbf_deri(x, height, sigma, tau, offset_rbf):
    return (rbf(x, height, sigma, tau, offset_rbf)  - offset_rbf) * (-sigma * 2 * (x - tau))

def rbf_u(x, height, sigma, tau, offset_rbf, beta, gamma, intercept):
    return (rbf_deri(x, height, sigma, tau, offset_rbf) + gamma * rbf(x, height, sigma, tau, offset_rbf)) / beta + intercept

def plot_range(
    gene_name, 
    adata, 
    config_file=None, 
    save_fig=False, 
    show_legend=True,
    time_metric='latent_time',
    palette='tab20',
    size=20
):
    """
    Plotting function of phase portraits of individual genes.
    
    Args:
        gene_name (str): name of that gene to be illusrated, would extend to list of genes in next release
        adata (AnnData)
        config_file (.Config class): configuration file used for velocity estimation
        save_fig (bool): if True, save fig, default False
        
        show_ax (bool)
        show_legend (bool)
        time_metric (str): inferred cell time, default 'latent_time'
    """

    if config_file == None:
        raise ValueError('Please set attribute `config_file`')

    if time_metric == 'latent_time':
        if 'latent_time' not in adata.obs.columns:
            scv.tl.latent_time(adata, min_likelihood=None)

    gene_name = gene_name if type(gene_name) == list else [gene_name]

    for gn in gene_name:
        fig, axes = plt.subplots(
                nrows=1,
                ncols=3, 
                figsize=(18, 4)
        )
        gdata = adata[:, gn]

        boundary = (gdata.var.fit_t.values - 3 * (1 / np.sqrt(2 * np.exp(gdata.var.fit_a.values))), 
                    gdata.var.fit_t.values + 3 * (1 / np.sqrt(2 * np.exp(gdata.var.fit_a.values))))
        
        t_one = np.linspace(0, 1, 1000)
        t_boundary = np.linspace(boundary[0], boundary[1], 2000)

        spre = np.squeeze(rbf(t_boundary, gdata.var.fit_h.values, gdata.var.fit_a.values, gdata.var.fit_t.values, gdata.var.fit_offset.values))
        sone = np.squeeze(rbf(t_one, gdata.var.fit_h.values, gdata.var.fit_a.values, gdata.var.fit_t.values, gdata.var.fit_offset.values))

        upre = np.squeeze(rbf_u(t_boundary, gdata.var.fit_h.values, gdata.var.fit_a.values, gdata.var.fit_t.values, gdata.var.fit_offset.values, gdata.var.fit_beta.values, gdata.var.fit_gamma.values, gdata.var.fit_intercept.values))
        uone = np.squeeze(rbf_u(t_one, gdata.var.fit_h.values, gdata.var.fit_a.values, gdata.var.fit_t.values, gdata.var.fit_offset.values, gdata.var.fit_beta.values, gdata.var.fit_gamma.values, gdata.var.fit_intercept.values))

        g1 = sns.scatterplot(x=np.squeeze(gdata.layers['Ms']), 
                            y=np.squeeze(gdata.layers['Mu']), 
                            s=size, hue=adata.obs[adata.uns['label']], 
                            palette=palette, ax=axes[0])
        axes[0].plot(spre, upre, color='lightgrey', linewidth=2, label='Predicted Curve')
        axes[0].plot(sone, uone, color='black', linewidth=2, label='Predicted Curve Time 0-1')
        axes[0].set_xlabel('Spliced Reads')
        axes[0].set_ylabel('Unspliced Reads')

        axes[0].set_xlim([-0.005 if gdata.layers['Ms'].min() < 1
                else gdata.layers['Ms'].min() * 0.95, 
                gdata.layers['Ms'].max() * 1.05])
        axes[0].set_ylim([-0.005 if gdata.layers['Mu'].min() < 1
                else gdata.layers['Mu'].min() * 0.95, 
                gdata.layers['Mu'].max() * 1.05])

        g2 = sns.scatterplot(x=np.squeeze(adata.obs[time_metric].values), 
                            y=np.squeeze(gdata.layers['Ms']), 
                            s=size, hue=adata.obs[adata.uns['label']], 
                            palette=palette, ax=axes[1])
        sns.lineplot(x=t_one, y=sone, color='black', linewidth=2, ax=axes[1])

        axes[1].set_xlabel('Inferred Cell Time')
        axes[1].set_ylabel('Spliced')

        g3 = sns.scatterplot(x=np.squeeze(adata.obs[time_metric].values), 
                            y=np.squeeze(gdata.layers['Mu']), 
                            s=size, hue=adata.obs[adata.uns['label']], 
                            palette=palette, ax=axes[2])
        sns.lineplot(x=t_one, y=uone, color='black', linewidth=2, ax=axes[2])

        axes[2].set_xlabel('Inferred Cell Time')
        axes[2].set_ylabel('Unspliced')

        if not show_legend:
            g1.get_legend().remove()
            g2.get_legend().remove()
            g3.get_legend().remove()

        axes[1].set_title(gn, fontsize=12)
        plt.show()

        if save_fig:
            plt.savefig(os.path.join(f'GM_{gn}.png'), dpi=300, bbox_inches='tight')

def plot_loss(iter, loss, thres=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = range(iter + 1)

    subiter, subloss = x[800:thres - 1], loss[800:thres - 1]
    axes[0].plot(subiter, subloss)
    axes[0].set_title('Iter # from 800 to cutoff')
    axes[0].set_ylabel('Euclidean Loss')

    # subiter, subloss = x[int(iter / 2):], loss[int(iter / 2):]
    # axes[1].plot(subiter, subloss)
    # axes[1].set_title('Iter # from 1/2 of maximum')

    subiter, subloss = x[int(thres * 1.01):], loss[int(thres * 1.01):]
    axes[1].plot(subiter, subloss)
    axes[1].set_title('Iter # from cutoff to terminated state')

    plt.show()
    plt.close()

def plot_reverse_tran_scatter(adata):
    sns.scatterplot(x='rbf_r2', y='qua_r2', data=adata.var.loc[adata.var['velocity_genes'] == True])
    plt.axline((0, 0), (0.5, 0.5), color='r')
        
    plt.title(f'$R^2$ comparison of RBF and Quadratic model')
    plt.show()
    plt.close()


def gene_weights_plots(genes, adata, adata_atac):
    c_ = np.where(np.log(adata.varm["fit_region_weights"])!=0)[1]
    r_ = np.where(np.log(adata.varm["fit_region_weights"])!=0)[0]

    for gene in genes:
        gene_name = gene
        
        adata.var[adata.var_names == gene_name]
        gene_number = np.where(adata.var_names == gene_name)[0][0]
        r_g = adata.varm["fit_region_weights"][gene_number,c_[r_ == gene_number]]
        adata_atac.var["chromStart"][c_[r_ == gene_number]]
            
        
        distance_regions = adata_atac.var["chromStart"][c_[r_ == gene_number]].values
        gene_end = adata.var[adata.var_names == gene_name]["chromEnd"].values[0]
        gene_start = adata.var[adata.var_names == gene_name]["chromStart"].values[0]

        plt.figure(figsize=(4, 3))
        plt.axvline(x=0, ymin=0, ymax=10,color='green', linestyle='dotted')
        plt.axvline(x=gene_end - gene_start, ymin=0, ymax=10,color='red', linestyle='dotted')
        
        # plt.xlim([-10000,20000])
        plt.xlabel('distance')
        plt.ylabel('$w^{r}$')
        plt.title(gene)
        plot = sns.scatterplot(x= distance_regions - gene_start, y=r_g )
        
       


def get_scatter_markers(n):
    # Available marker styles in matplotlib
    markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']
    
    # If n is larger than the number of available markers, repeat the markers
    return markers * (n // len(markers)) + markers[:n % len(markers)]


def region_kinetic_plot(adata,adata_atac,gene_name, time, average_intervals,
                        average_intervals_unspliced, B,
                        save_fig = False, path = " "):
    
    w = np.multiply(adata.varm["fit_region_weights"],B.T )
    c_ = np.where(w!=0)[1]
    r_ = np.where(w!=0)[0]

  
    adata.var[adata.var_names == gene_name]
    gene_number = np.where(adata.var_names == gene_name)[0][0]
    r_g = adata.varm["fit_region_weights"][gene_number,c_[r_ == gene_number]]
    adata_atac.var["chromStart"][c_[r_ == gene_number]]
        
    
    distance_regions = adata_atac.var["chromStart"][c_[r_ == gene_number]].values
    gene_end = adata.var[adata.var_names == gene_name]["chromEnd"].values[0]
    gene_start = adata.var[adata.var_names == gene_name]["chromStart"].values[0]
    
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    colors = plt.get_cmap('Set2').colors
    markers_list = get_scatter_markers(time.shape[0])
    
    for i in np.arange(0,time.shape[0]):
        
        plot = sns.scatterplot(y=average_intervals[i], x =time[i] ,alpha=1,
                               label ="d = "+ str((distance_regions - gene_start)[i]), 
                                color = colors[i], marker = markers_list[i], ax = ax1 )


       

    plt.ylabel(f'$c_r^g$')
    plt.xlabel("time")
    
    plot.spines['top'].set_visible(False)
   
    plt.title(gene_name)
    
    ax2 = ax1.twinx()
    plot = sns.lineplot(y=average_intervals_unspliced, 
                        x =time[i] ,alpha=1, marker='o',label = 'u', ax = ax2
                        ,legend= False)
    for line in plot.get_lines():
        line.set_linestyle('--') 

    plot.spines['top'].set_visible(False)
    plt.ylabel('u')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(lines+lines2 , labels+labels2 , loc='best', bbox_to_anchor=(1.3, 1))

    plt.tight_layout() 

    if save_fig:
        plt.savefig(path+str(gene_name)+'_kinetic_plot.png', dpi=300)
        plt.savefig(path+str(gene_name)+'_kinetic_plot.pdf', dpi=300)


    plt.figure(figsize=(4, 2))

    plt.axvline(x=0, ymin=0, ymax=10,color='green', linestyle='dotted', alpha = 1, label = 'gene begin')
    plt.axvline(x=gene_end - gene_start, ymin=0, ymax=10,color='blue',
                 linestyle='dotted', alpha = 0.5, label = 'gene end')
    
    for i in np.arange(0,time.shape[0]):
      
      plot = sns.scatterplot( x= [(distance_regions - gene_start)[i]],
                         y=[r_g[i]], color = colors[i], marker = markers_list[i],alpha=1)
    plt.ylabel(f'$w_r^g$')
    plt.xlabel('distance')

    plt.xlim([-10000,10000])
    
    plt.legend(loc='best',bbox_to_anchor=(1, 1), frameon=False)
    plot.spines['top'].set_visible(False)
    plot.spines['right'].set_visible(False)
    plt.tight_layout() 

    if save_fig:
        plt.savefig(path+str(gene_name)+'_regions_weights.png', dpi=300)
        plt.savefig(path+str(gene_name)+'_regions_weights.pdf', dpi=300)
     
    plt.show()
    

def scatter_plot(adata,
                 genes,
                 by='us',
                 color_by="celltype",
                 n_cols=5,
                 axis_on=True,
                 frame_on=True,
                 
                 title_more_info=False,
                 
                 downsample=1,
                 figsize=None,
                 pointsize=2,
                
                 cmap='coolwarm',
                 view_3d_elev=None,
                 view_3d_azim=None,
                 full_name=False
                 ):
   
        from pandas.api.types import is_numeric_dtype, is_categorical_dtype
        if by not in ['us', 'cu']:
            raise ValueError("'by' argument must be one of ['us', 'cu']")
       
        elif by == 'us' and color_by == 'c':
            types = None
        elif color_by in adata.obs and is_numeric_dtype(adata.obs[color_by]):
            types = None
            colors = adata.obs[color_by].values
        elif color_by in adata.obs and is_categorical_dtype(adata.obs[color_by]) \
                and color_by+'_colors' in adata.uns.keys():
            types = adata.obs[color_by].cat.categories
            colors = adata.uns[f'{color_by}_colors']
        else:
            raise ValueError('Currently, color key must be a single string of '
                            'either numerical or categorical available in adata'
                            ' obs, and the colors of categories can be found in'
                            ' adata uns.')

       
       
        downsample = np.clip(int(downsample), 1, 10)
        genes = np.array(genes)
        missing_genes = genes[~np.isin(genes, adata.var_names)]
        if len(missing_genes) > 0:
            print(f'{missing_genes} not found')
        genes = genes[np.isin(genes, adata.var_names)]
        gn = len(genes)
        if gn == 0:
            return
        if gn < n_cols:
            n_cols = gn
        
        fig, axs = plt.subplots(-(-gn // n_cols), n_cols, squeeze=False,
                                    figsize=(2.7*n_cols, 2.4*(-(-gn // n_cols)))
                                    if figsize is None else figsize)
        fig.patch.set_facecolor('white')
        count = 0
        for gene in genes:
            u = adata[:, gene].layers['Mu'].copy() if 'Mu' in adata.layers \
                else adata[:, gene].layers['unspliced'].copy()
            s = adata[:, gene].layers['Ms'].copy() if 'Ms' in adata.layers \
                else adata[:, gene].layers['spliced'].copy()
            u = u.A if sparse.issparse(u) else u
            s = s.A if sparse.issparse(s) else s
            u, s = np.ravel(u), np.ravel(s)
            
            if 'ATAC' in adata.layers.keys():
                c = adata[:, gene].layers['ATAC'].copy()
                c = c.A if sparse.issparse(c) else c
                c = np.ravel(c)
            elif 'Mc' in adata.layers.keys():
                c = adata[:, gene].layers['Mc'].copy()
                c = c.A if sparse.issparse(c) else c
                c = np.ravel(c)

            

            row = count // n_cols
            col = count % n_cols
            ax = axs[row, col]
            if types is not None:
                for i in range(len(types)):
                    
                    filt = adata.obs[color_by] == types[i]
                    filt = np.ravel(filt)
                    if by == 'us':
                       
                            ax.scatter(s[filt][::downsample],
                                    u[filt][::downsample], s=pointsize,
                                    c=colors[i], alpha=0.7)
                    elif by == 'cu':
                      
                       
                            ax.scatter(u[filt][::downsample],
                                    c[filt][::downsample], s=pointsize,
                                    c=colors[i], alpha=0.7)
                    else:
                        
                            ax.scatter(s[filt][::downsample],
                                    u[filt][::downsample],
                                    c[filt][::downsample], s=pointsize,
                                    c=colors[i], alpha=0.7)
            elif color_by == 'c':
                if 'velo_s_params' in adata.uns.keys() and \
                        'outlier' in adata.uns['velo_s_params']:
                    outlier = adata.uns['velo_s_params']['outlier']
                else:
                    outlier = 99.8
                non_zero = (u > 0) & (s > 0) & (c > 0)
                non_outlier = u < np.percentile(u, outlier)
                non_outlier &= s < np.percentile(s, outlier)
                non_outlier &= c < np.percentile(c, outlier)
                c -= np.min(c)
                c /= np.max(c)
                
                ax.scatter(s[non_zero & non_outlier][::downsample],
                            u[non_zero & non_outlier][::downsample],
                            s=pointsize,
                            c=np.log1p(c[non_zero & non_outlier][::downsample]),
                            alpha=0.8, cmap=cmap)
            else:
                if by == 'us':
                   
                        ax.scatter(s[::downsample], u[::downsample], s=pointsize,
                                c=colors[::downsample], alpha=0.7, cmap=cmap)
                elif by == 'cu':
                    
                        ax.scatter(u[::downsample], c[::downsample], s=pointsize,
                                c=colors[::downsample], alpha=0.7, cmap=cmap)
                

            

            
            title = gene
            if title_more_info:
                if 'fit_model' in adata.var:
                    title += f" M{int(adata[:,gene].var['fit_model'].values[0])}"
                if 'fit_direction' in adata.var:
                    title += f" {adata[:,gene].var['fit_direction'].values[0]}"
                if 'fit_likelihood' in adata.var \
                        and not np.all(adata.var['fit_likelihood'].values == -1):
                    title += " "
                    f"{adata[:,gene].var['fit_likelihood'].values[0]:.3g}"
            ax.set_title(f'{title}', fontsize=11)
            if by == 'us':
                ax.set_xlabel('spliced' if full_name else 's')
                ax.set_ylabel('unspliced' if full_name else 'u')
            elif by == 'cu':
                ax.set_xlabel('unspliced' if full_name else 'u')
                ax.set_ylabel('chromatin' if full_name else 'c')
       
            if by in ['us', 'cu']:
                if not axis_on:
                    ax.xaxis.set_ticks_position('none')
                    ax.yaxis.set_ticks_position('none')
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                if not frame_on:
                    ax.xaxis.set_ticks_position('none')
                    ax.yaxis.set_ticks_position('none')
                    ax.set_frame_on(False)
            
            count += 1
        for i in range(col+1, n_cols):
            fig.delaxes(axs[row, i])
        fig.tight_layout()

   
