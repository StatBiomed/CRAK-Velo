#%%
import matplotlib.pyplot as plt
import scvelo as scv
import seaborn as sns
import numpy as np
import os

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