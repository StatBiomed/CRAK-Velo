import numpy as np
import os
import pandas as pd
from .model import lagrange
from .utils import make_dense, get_weight, R2
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(42)

class Velocity:
    def __init__(
        self,
        adata=None,
        min_ratio=0.01,
        min_r2=0.01,
        perc=[5, 95],
        vkey='velocity',
        config=None
    ):
        self.adata = adata
        self.vkey = vkey

        self.Ms = adata.layers["spliced"] if config.USE_RAW else adata.layers["Ms"].copy()
        self.Mu = adata.layers["unspliced"] if config.USE_RAW else adata.layers["Mu"].copy()
        self.Ms, self.Mu = make_dense(self.Ms), make_dense(self.Mu)

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

        if self.config.R2_ADJUST:
            Ms, Mu = self.Ms, self.Mu

        self.gene_index = variable.index
        self.gamma_ref = np.sum(Mu * Ms, axis=0) / np.sum(Ms * Ms, axis=0)
        self.residual_scale = self.Mu - self.gamma_ref * self.Ms
        self.r2 = R2(self.residual_scale, total=self.Mu - np.mean(self.Mu, axis=0))

        self.velocity_genes = np.ones(Ms.shape[1], dtype=bool)

        if type(self.config.VGENES) == list:
            temp = []
            for gene in variable.index:
                if gene in self.config.VGENES:
                    temp.append(True)
                else:
                    temp.append(False)
                    
            self.velocity_genes = np.array(temp)
            self.adata.var['velocity_gamma'] = self.gamma_ref

        elif self.config.VGENES == 'raws':
            self.velocity_genes = np.ones(Ms.shape[1], dtype=np.bool)

        elif self.config.VGENES == 'offset':
            self.fit_linear(self.Ms, self.Mu)

        elif self.config.VGENES == 'basic':
            self.velocity_genes = (
                (self.r2 > self.min_r2)
                & (self.r2 < 0.95)
                & (self.gamma_quantile > self.min_ratio)
                & (self.gamma_ref > self.min_ratio)
                & (np.max(self.Ms > 0, axis=0) > 0)
                & (np.max(self.Mu > 0, axis=0) > 0)
            )
            print (f'# of velocity genes {self.velocity_genes.sum()} (Criterion: positive regression coefficient between un/spliced counts)')
            
            if self.config.R2_ADJUST:
                lb, ub = np.nanpercentile(self.scaling, [10, 90])
                self.velocity_genes = (
                    self.velocity_genes
                    & (self.scaling > np.min([lb, 0.03]))
                    & (self.scaling < np.max([ub, 3]))
                )
            print (f'# of velocity genes {self.velocity_genes.sum()} (Criterion: std of un/spliced reads should be moderate, w/o extreme values)')

            self.adata.var['velocity_gamma'] = self.gamma_ref
            self.adata.var['velocity_r2'] = self.r2

        else:
            raise ValueError('Plase specify the correct self.VGENES in configuration file')

        if True:
            self.init_weights()
            self.velocity_genes = self.velocity_genes & (self.nobs > 0.05 * Ms.shape[1])

        self.adata.var['scaling'] = self.scaling
        self.adata.var['velocity_genes'] = self.velocity_genes
        
        if np.sum(self.velocity_genes) < 2:
            print ('---> Low signal in splicing dynamics.')

    def init_weights(self):
        nonzero_s, nonzero_u = self.Ms > 0, self.Mu > 0
        weights = np.array(nonzero_s & nonzero_u, dtype=bool)
        self.nobs = np.sum(weights, axis=0)

    def fit_linear(self, Ms, Mu):
        from sklearn import linear_model
        from sklearn.metrics import r2_score

        index = self.adata.var.index
        linear = pd.DataFrame(index=index, data=0, dtype=np.float32, 
            columns=['coef', 'inter', 'r2'])

        reg = linear_model.LinearRegression()
        for col in range(Ms.shape[1]):
            sobs = np.reshape(Ms[:, col], (-1, 1))
            uobs = np.reshape(Mu[:, col], (-1, 1))

            reg.fit(sobs, uobs)
            u_pred = reg.predict(sobs)

            linear.loc[index[col], 'coef'] = float(reg.coef_.squeeze())
            linear.loc[index[col], 'inter'] = float(reg.intercept_.squeeze())
            linear.loc[index[col], 'r2'] = r2_score(uobs, u_pred)

        self.adata.var['velocity_inter'] = np.array(linear['inter'].values)
        self.adata.var['velocity_gamma'] = np.array(linear['coef'].values)
        self.adata.var['velocity_r2'] = np.array(linear['r2'].values)
        self.gamma_ref = np.array(linear['coef'].values)
        self.r2 = np.array(linear['r2'].values)

        self.velocity_genes = (
            self.velocity_genes
            & (self.r2 > self.min_r2)
            & (self.r2 < 0.95)
            & (np.array(linear['coef'].values) > self.min_ratio)
            & (np.max(self.Ms > 0, axis=0) > 0)
            & (np.max(self.Mu > 0, axis=0) > 0)
        )
        print (f'# of velocity genes {self.velocity_genes.sum()} (Criterion: positive regression coefficient between un/spliced counts)')

        lb, ub = np.nanpercentile(self.scaling, [10, 90])
        self.velocity_genes = (
            self.velocity_genes
            & (self.scaling > np.min([lb, 0.03]))
            & (self.scaling < np.max([ub, 3]))
        )
        print (f'# of velocity genes {self.velocity_genes.sum()} (Criterion: std of un/spliced reads should be moderate, w/o extreme values)')

    def fit_curve(self, adata, idx, Ms_scale, Mu_scale):
        physical_devices = tf.config.list_physical_devices('GPU')

        if len(physical_devices) == 0 or self.config.GPU == -1:
            tf.config.set_visible_devices([], 'GPU')
            device = '/cpu:0'
            print ('No GPU device has been detected. Switch to CPU mode.')

        else:
            assert self.config.GPU < len(physical_devices), 'Please specify the correct GPU card.'
            tf.config.set_visible_devices(physical_devices[self.config.GPU], 'GPU')

            os.environ["CUDA_VISIBLE_DEVICES"] = f'{self.config.GPU}'
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)

            device = f'/gpu:{self.config.GPU}'
            print (f'Using GPU card: {self.config.GPU}')

        with tf.device(device):
            residual, adata = lagrange(
                adata, idx=idx,
                Ms=Ms_scale, Mu=Mu_scale, 
                config=self.config
            )

        return residual, adata

    def fit_velo_genes(self):
        idx = self.velocity_genes
        print (f'# of velocity genes {idx.sum()} (Criterion: genes have reads in more than 5% of total cells)')

        if self.config.RESCALE_DATA:
            Ms_scale, Mu_scale = self.Ms, self.Mu / self.scaling
        else:
            Ms_scale, Mu_scale = self.Ms, self.Mu

        residual, adata = self.fit_curve(self.adata, idx, Ms_scale, Mu_scale)
        adata.layers[self.vkey] = residual
        
        if self.config.FIT_OPTION == '1':
            adata.obs['latent_time'] = adata.obs['latent_time_gm']
            del adata.obs['latent_time_gm']

        return adata