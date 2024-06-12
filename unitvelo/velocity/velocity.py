import numpy as np
import os
import pandas as pd
from model import lagrange
from utils import make_dense, get_weight, R2
import tensorflow as tf
from sklearn import linear_model
from sklearn.metrics import r2_score

class Velocity:
    def __init__(
        self,
        adata=None,
        #adata_atac = None,
        #df_rg_intersection = None
        logger=None,
        min_ratio=0.01,
        min_r2=0.01,
        perc=[5, 95],
        config=None
    ):
        self.adata = adata
        #self_adata_atac = adata_atac
        #self.df_rg_intersection = df_rg_intersection
        self.logger = logger

        self.Ms = adata.layers["spliced"] if config['preprocessing']['use_raw'] else adata.layers["Ms"].copy()
        self.Mu = adata.layers["unspliced"] if config['preprocessing']['use_raw'] else adata.layers["Mu"].copy()
        self.Ms, self.Mu = make_dense(self.Ms), make_dense(self.Mu)
        ##self.Matac = self.adata_atac.X ##(sparse matrix)
        #self.M_acc = self.adata_atac.obsm["cisTopic"].numpy()
        
        self.min_r2 = min_r2
        self.min_ratio = min_ratio
        
        n_obs, n_vars = self.Ms.shape
        self.gamma = np.zeros(n_vars, dtype=np.float32)
        self.r2 = np.zeros(n_vars, dtype=np.float32)
        self.velocity_genes = np.ones(n_vars, dtype=bool)
        self.residual_scale = np.zeros([n_obs, n_vars], dtype=np.float32)
    
        self.config = config

    def get_velo_genes(self):
        variable = self.adata.var

        if variable.index[0].startswith('ENSMUSG'):
            variable.index = variable['gene']
            variable.index.name = 'index' 
        
        weights = get_weight(self.Ms, self.Mu, perc=95)
        Ms, Mu = weights * self.Ms, weights * self.Mu

        self.gamma_quantile = np.sum(Mu * Ms, axis=0) / np.sum(Ms * Ms, axis=0)
        self.scaling = np.std(self.Mu, axis=0) / np.std(self.Ms, axis=0)
        self.adata.layers['Mu_scale'] = self.Mu / self.scaling

        if self.config['velocity_genes']['r2_adjust']:
            Ms, Mu = self.Ms, self.Mu

        self.gene_index = variable.index
        self.gamma_ref = np.sum(Mu * Ms, axis=0) / np.sum(Ms * Ms, axis=0)
        self.residual_scale = self.Mu - self.gamma_ref * self.Ms
        self.r2 = R2(self.residual_scale, total=self.Mu - np.mean(self.Mu, axis=0))

        self.velocity_genes = np.ones(Ms.shape[1], dtype=bool)

        if isinstance(self.config['velocity_genes']['vgenes'], list):           
            self.velocity_genes = np.isin(variable.index, self.config['velocity_genes']['vgenes'])
            self.adata.var['velocity_gamma'] = self.gamma_ref

        elif self.config['velocity_genes']['vgenes'] == 'raws':
            pass

        elif self.config['velocity_genes']['vgenes'] == 'offset':
            self.fit_linear(self.Ms, self.Mu)

        elif self.config['velocity_genes']['vgenes'] == 'basic':
            self.velocity_genes = (
                (self.r2 > self.min_r2)
                & (self.r2 < 0.95)
                & (self.gamma_quantile > self.min_ratio)
                & (self.gamma_ref > self.min_ratio)
                & (np.max(self.Ms > 0, axis=0) > 0)
                & (np.max(self.Mu > 0, axis=0) > 0)
            )
            self.logger.info(f'# of velocity genes {self.velocity_genes.sum()} (Criterion: positive regression coefficient between un/spliced counts)')
            
            if self.config['velocity_genes']['r2_adjust']:
                lb, ub = np.nanpercentile(self.scaling, [10, 90])
                self.velocity_genes = (
                    self.velocity_genes
                    & (self.scaling > np.min([lb, 0.03]))
                    & (self.scaling < np.max([ub, 3]))
                )
            self.logger.info(f'# of velocity genes {self.velocity_genes.sum()} (Criterion: std of un/spliced reads should be moderate, w/o extreme values)\n')

            self.adata.var['velocity_gamma'] = self.gamma_ref
            self.adata.var['velocity_r2'] = self.r2

        else:
            raise ValueError('Plase specify the correct vgenes mode in the configuration file')

        self.init_weights()
        self.velocity_genes = self.velocity_genes & (self.nobs > 0.05 * Ms.shape[1])
        self.logger.info(f'# of velocity genes {self.velocity_genes.sum()} (Criterion: genes have reads in more than 5% of total cells)\n')

        self.adata.var['scaling'] = self.scaling
        self.adata.var['velocity_genes'] = self.velocity_genes
        
        if np.sum(self.velocity_genes) < 2:
            self.logger.info('Low signal in splicing dynamics.')

    def init_weights(self):
        nonzero_s, nonzero_u = self.Ms > 0, self.Mu > 0
        weights = np.array(nonzero_s & nonzero_u, dtype=bool)
        self.nobs = np.sum(weights, axis=0)

    
       
        
    def fit_linear(self, Ms, Mu):
        index = self.adata.var.index
        linear_results = pd.DataFrame(
            index=index, data=0, dtype=np.float32, 
            columns=['coef', 'inter', 'r2']
        )

        reg = linear_model.LinearRegression()
        for i, gene in enumerate(index):
            sobs = Ms[:, i].reshape(-1, 1)
            uobs = Mu[:, i].reshape(-1, 1)

            reg.fit(sobs, uobs)
            u_pred = reg.predict(sobs)

            linear_results.loc[gene, 'coef'] = reg.coef_.squeeze()
            linear_results.loc[gene, 'inter'] = reg.intercept_.squeeze()
            linear_results.loc[gene, 'r2'] = r2_score(uobs, u_pred)

        self.adata.var['velocity_inter'] = linear_results['inter'].values
        self.adata.var['velocity_gamma'] = linear_results['coef'].values
        self.adata.var['velocity_r2'] = linear_results['r2'].values

        self.gamma_ref = linear_results['coef'].values
        self.r2 = linear_results['r2'].values

        self.velocity_genes = (
            self.velocity_genes
            & (self.r2 > self.min_r2)
            & (self.r2 < 0.95)
            & (linear_results['coef'].values > self.min_ratio)
            & (np.max(self.Ms > 0, axis=0) > 0)
            & (np.max(self.Mu > 0, axis=0) > 0)
        )
        self.logger.info(f'# of velocity genes {self.velocity_genes.sum()} (Criterion: positive regression coefficient between un/spliced counts)')

        lb, ub = np.nanpercentile(self.scaling, [10, 90])
        self.velocity_genes = (
            self.velocity_genes
            & (self.scaling > np.min([lb, 0.03]))
            & (self.scaling < np.max([ub, 3]))
        )
        self.logger.info(f'# of velocity genes {self.velocity_genes.sum()} (Criterion: std of un/spliced reads should be moderate, w/o extreme values)\n')

    def _prepare_device(self, gpu_id):
        available_gpus = tf.config.list_physical_devices('GPU')
        n_gpu = len(available_gpus)

        if n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine, traning will be performed on CPU.")
            return '/cpu:0'

        assert self.config['system']['gpu_id'] < n_gpu, 'Please specify the correct GPU card.'
        tf.config.set_visible_devices(available_gpus[gpu_id], 'GPU')

        os.environ["CUDA_VISIBLE_DEVICES"] = f'{gpu_id}'
        
        for gpu in available_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        self.logger.info(f'Using GPU card: {gpu_id}')
        return f'/gpu:{gpu_id}'

    def fit_curve(self, adata, Ms_scale, Mu_scale):
        self.device = self._prepare_device(self.config['system']['gpu_id'])

        with tf.device(self.device):
            return lagrange(
                adata, 
                idx=self.velocity_genes,
                Ms=Ms_scale, 
                Mu=Mu_scale, 
                config=self.config,
                logger=self.logger
            )

    def fit_velo_genes(self):
        if self.config['preprocessing']['rescale_data']:
            Ms_scale, Mu_scale = self.Ms, self.Mu / self.scaling
        else:
            Ms_scale, Mu_scale = self.Ms, self.Mu

        return self.fit_curve(self.adata, Ms_scale, Mu_scale)