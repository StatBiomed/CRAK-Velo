import tensorflow as tf
import numpy as np
from tqdm import tqdm
from .optimize_utils import Model_Utils
from utils import min_max
from packaging import version
import os
import scanpy as sc

exp = tf.math.exp
log = tf.math.log
sum = tf.math.reduce_sum
mean = tf.math.reduce_mean
sqrt = tf.math.sqrt
abs = tf.math.abs
square = tf.math.square
pow = tf.math.pow
std = tf.math.reduce_std
var = tf.math.reduce_variance

class Recover_Paras(Model_Utils):
    def __init__(
        self,
        adata, 
        M_acc, 
        B,
        Ms,
        Mu,
        idx=None,
        config=None,
        logger=None
    ):
        super().__init__(
            adata=adata,
            M_acc=M_acc,
            B=B, 
            Ms=Ms,
            Mu=Mu,
            config=config,
            logger=logger
        )

        self.idx = idx
        self.scaling = adata.var['scaling'].values
        self.default_pars_names = ['gamma', 'beta', 'offset', 'a', 't', 'h', 'intercept','region_weights', 'etta']
        # self.default_pars_names += ['region_weights', 'etta']

        self.init_vars()
        self.init_weights()

        self.t_cell = self.compute_cell_time(args=None)
        self.pi = tf.constant(np.pi, dtype=tf.float32)

    def predict_cell_time(self, args, x):
        s_predict, u_predict = self.get_s_u(args, x)
        s_predict = tf.expand_dims(s_predict, axis=0) # 1 3000 d
        u_predict = tf.expand_dims(u_predict, axis=0)

        Mu = tf.expand_dims(self.Mu, axis=1) # n 1 d
        Ms = tf.expand_dims(self.Ms, axis=1)

        t_cell = self.match_time(Ms, Mu, s_predict, u_predict, x.numpy())

        if self.config['fitting_option']['aggregrate_t']:
            t_cell = tf.reshape(t_cell, (-1, 1))
            t_cell = tf.broadcast_to(t_cell, self.adata.shape)
        
        return t_cell

    def init_cell_time_from_gene_counts(self):
        self.logger.info('Use gene counts for cell time initialization.')

        self.adata.obs['gcount'] = np.sum(self.adata.X.todense() > 0, axis=1)
        g_time = 1 - min_max(self.adata.obs.groupby(self.config['cluster_name'])['gcount'].mean())

        for id in list(g_time.index):
            self.adata.obs.loc[self.adata.obs[self.config['cluster_name']] == id, 'gcount'] = g_time[id]

        return tf.cast(
            tf.broadcast_to(
                self.adata.obs['gcount'].values.reshape(-1, 1), 
                self.adata.shape
            ), 
            tf.float32
        )
    
    def init_cell_time_from_diffusion_pseudotime(self):
        self.logger.info('Use diffusion pseudotime for cell time initialization.')

        sc.tl.diffmap(self.adata)
        self.adata.uns['iroot'] = np.flatnonzero(
                self.adata.obs[self.config['cluster_name']] == self.config['cell_initialization']['iroot']
            )[0]
        sc.tl.dpt(self.adata)
        
        return tf.cast(
            tf.broadcast_to(
                self.adata.obs['dpt_pseudotime'].values.reshape(-1, 1), 
                self.adata.shape
            ), 
            tf.float32
        )

    def compute_cell_time(self, args=None):
        if args is not None:
            x = self.init_time((0, 1), (3000, self.adata.n_vars))
            t_cell = self.predict_cell_time(args, x)

        else:
            t_cell = self.init_time((0, 1), self.adata.shape)

            if self.config['cell_initialization']['iroot'] == 'gcount':
                t_cell = self.init_cell_time_from_gene_counts()

            elif self.config['cell_initialization']['iroot'] in self.adata.obs[self.config['cluster_name']].values:
                t_cell = self.init_cell_time_from_diffusion_pseudotime()
        
            else:
                self.logger.warning("Invalid cell initialization method. Using default initialization.")

        return t_cell

    def get_regularization_errors(self, args):
        self.s_r2 = self.s_r2 + \
            std(self.Ms, axis=0) * self.config['regularization']['reg_times'] * \
            exp(-square(args[4] - 0.5) / self.config['regularization']['reg_scale'])

    def get_squared_errors(self, args, t_cell, iter):
        self.s_func, self.u_func = self.get_s_u(args, t_cell)
        self.udiff, self.sdiff = self.Mu - self.u_func, self.Ms - self.s_func
        self.u_deri_func = self.get_u_deri(args, t_cell)
        
        latent_time_ = self.get_interim_t(self.t_cell, self.adata.var['velocity_genes'].values)
        latent_time_ = min_max(latent_time_[:, 0]) 

        self.u_deri_atac = self.compute_u_deri_atac(args, latent_time_)
        self.u_deri_diff = self.u_deri_func - self.u_deri_atac
       
        self.u_r2 = square(self.udiff)
        self.s_r2 = square(self.sdiff)
        self.u_deri_r2 = square(self.u_deri_diff)
        
        if (self.config['fitting_option']['mode'] == 1) & \
            (iter > int(0.9 * self.config['base_trainer']['epochs'])) & \
            self.config['regularization']['reg_loss']:

            self.get_regularization_errors(args)

    def get_variances(self):
        #! convert for self.varu to account for scaling in pre-processing
        self.vars = mean(self.s_r2, axis=0) \
            - square(mean(tf.math.sign(self.sdiff) * sqrt(self.s_r2), axis=0))
        self.varu = mean(self.u_r2 * square(self.scaling), axis=0) \
            - square(mean(tf.math.sign(self.udiff) * sqrt(self.u_r2) * self.scaling, axis=0))

    def get_log_likelihood(self):
        self.u_log_likeli = \
            - (self.Mu.shape[0] / 2) * log(2 * self.pi * self.varu) \
            - sum(self.u_r2 * square(self.scaling), axis=0) / (2 * self.varu) 
        self.s_log_likeli = \
            - (self.Ms.shape[0] / 2) * log(2 * self.pi * self.vars) \
            - sum(self.s_r2, axis=0) / (2 * self.vars) 

    def finalize_loss(self, iter, s_r2, u_r2, u_deri_r2):
        ngenes = self.Ms.shape[1]
        nregions = self.M_acc.shape[1] 
        reg_u_derr_loss = (ngenes)/(nregions)
        if self.config["base_trainer"]["loss_mode"] == 2:
            
            if iter < self.config['base_trainer']['epochs'] / 2:
                remain = iter % 400
                loss = s_r2  if remain < 200 else u_r2 

            else:
                loss = s_r2 + u_r2 +  reg_u_derr_loss*u_deri_r2
        
        if self.config["base_trainer"]["loss_mode"] == 1:
            
               
            if iter < self.config['base_trainer']['epochs'] /2:
                remain = iter % 400
                loss = s_r2 + reg_u_derr_loss*u_deri_r2    if remain < 200 else u_r2 + reg_u_derr_loss*u_deri_r2
            
           
            else:
                loss = s_r2 + u_r2 + reg_u_derr_loss*u_deri_r2
            


        return tf.where(tf.math.is_finite(loss), loss, 0)

    def compute_loss(self, args, t_cell, iter, progress_bar):
        self.get_squared_errors(args, t_cell, iter)
        self.get_variances()
        self.get_log_likelihood()

        error_1 = np.sum(sum(self.u_r2, axis=0).numpy()[self.idx]) / np.sum(self.idx)
        error_2 = np.sum(sum(self.s_r2, axis=0).numpy()[self.idx]) / np.sum(self.idx)
        error_3 = np.sum(sum( self.u_deri_r2, axis=0).numpy()[self.idx]) / np.sum(self.idx)
                          
        progress_bar.set_description(f'(Spliced): {error_2:.3f}, (Unspliced): {error_1:.3f}, (Unspliced_derrivative): {error_3:.3f}')
        
        return self.finalize_loss(
            iter,
            sum(self.s_r2, axis=0), 
            sum(self.u_r2, axis=0),
            sum(self.u_deri_r2, axis=0)
        )

    def init_optimizer(self):
        os.environ['TF_USE_LEGACY_KERAS'] = '1'

        if version.parse(tf.__version__) >= version.parse('2.11.0'):
            return tf.keras.optimizers.legacy.Adam(
                learning_rate=self.config['base_trainer']['learning_rate'], 
                amsgrad=True
            )
        
        else:
            return tf.keras.optimizers.Adam(
                learning_rate=self.config['base_trainer']['learning_rate'], 
                amsgrad=True
            )

    def update_and_store_results(self, args):
        self.t_cell = self.compute_cell_time(args=args)
        _ = self.get_fit_s(args, self.t_cell)

        s_derivative = self.get_s_deri(args, self.t_cell)
        u_derivative = self.get_u_deri(args, self.t_cell)
        u_derivative_atac = self.compute_u_deri_atac(args, self.t_cell[:, 0])
    
        self.post_utils(args)

        self.adata.var['velocity_genes'] = self.idx
        self.adata.layers['velocity'] = s_derivative.numpy()
        self.adata.layers['fit_t'] = self.t_cell.numpy() if self.config['fitting_option']['aggregrate_t'] else self.t_cell
        self.adata.layers['fit_t'][:, ~self.adata.var['velocity_genes'].values] = np.nan
        self.adata.layers['u_derrivative'] = u_derivative.numpy()
        self.adata.layers['u_atac'] = u_derivative_atac.numpy()

    def fit_likelihood(self):
        optimizer = self.init_optimizer()

        pre = tf.repeat(1e6, self.Ms.shape[1]) # (2000, )
        progress_bar = tqdm(range(self.config['base_trainer']['epochs']))

        for iter in progress_bar:
            with tf.GradientTape() as tape:                
                args = [
                    self.log_gamma, 
                    self.log_beta, 
                    self.offset, 
                    self.log_a, 
                    self.t, 
                    self.log_h, 
                    self.intercept,
                    self.log_region_weights,
                    self.log_etta
                ]
                obj = self.compute_loss(args, self.t_cell, iter, progress_bar)

            stop_cond = self.get_stop_cond(iter, pre, obj)

            if iter == self.config['base_trainer']['epochs'] - 1 or tf.math.reduce_all(stop_cond) == True:
                self.update_and_store_results(args)
                break
            
            args_to_optimize = self.get_opt_args(iter, args)
            gradients = tape.gradient(target=obj, sources=args_to_optimize)

            # convert gradients of variables with unused genes to 0
            convert = tf.cast(self.idx, tf.float32)            
            processed_grads = [g * convert for g in gradients]

            optimizer.apply_gradients(zip(processed_grads, args_to_optimize))
            args[7].assign(self.B * args[7])
            args[8].assign(self.B_genes_nr * args[8])
            pre = obj

            if iter > 0 and int(iter % 800) == 0:
                self.t_cell = self.compute_cell_time(args=args)

        latent_time = self.get_interim_t(self.t_cell, self.adata.var['velocity_genes'].values)
        self.adata.obs['latent_time'] = min_max(latent_time[:, 0])  
        return self.adata

    def post_utils(self, args):
        # Reshape un/spliced variance to (ngenes, ) and save
        self.adata.var['fit_vars'] = np.squeeze(self.vars)
        self.adata.var['fit_varu'] = np.squeeze(self.varu)

        # Save predicted parameters of RBF kernel to adata
        self.save_pars([item.numpy() for item in args])
        self.adata.var['fit_beta'] /= self.scaling
        self.adata.var['fit_intercept'] *= self.scaling

        self.adata.layers['Pred_s'] = self.s_func.numpy()
        self.adata.layers['Pred_u'] = self.u_func.numpy()

        r2_spliced = 1 - sum(self.s_r2, axis=0) / var(self.Ms, axis=0) / (self.adata.shape[0] - 1)
        r2_unspliced = 1 - sum(self.u_r2, axis=0) / var(self.Mu, axis=0) / (self.adata.shape[0] - 1)

        self.adata.var['fit_sr2'] = r2_spliced.numpy()
        self.adata.var['fit_ur2'] = r2_unspliced.numpy()

    def save_pars(self, paras):
        columns = ['a', 'h', 'gamma', 'beta']

        for i, name in enumerate(self.default_pars_names):
            
            if name == 'region_weights':
               self.adata.varm[f"fit_{name}"] = np.transpose(paras[i])
               self.adata.varm[f"fit_{name}"] = np.exp(self.adata.varm[f"fit_{name}"])
               #self.adata.varm[f"fit_{name}"] = self.B * self.adata.varm[f"fit_{name}"]
            else:
                self.adata.var[f"fit_{name}"] = np.transpose(np.squeeze(paras[i]))

                if name in columns:
                    self.adata.var[f"fit_{name}"] = np.exp(self.adata.var[f"fit_{name}"])
                if name == "etta":
                    self.adata.var[f"fit_{name}"] = np.exp(self.adata.var[f"fit_{name}"])
                    #self.adata.varm[f"fit_{name}"] = self.B_genes_nr * self.adata.varm[f"fit_{name}"]
def lagrange(
    adata,
    M_acc,
    B,
    idx=None,
    Ms=None,
    Mu=None,
    config=None,
    logger=None
):
    if len(set(adata.var_names)) != len(adata.var_names):
        adata.var_names_make_unique()

    model = Recover_Paras(
        adata,
        M_acc,
        B,
        Ms,
        Mu,
        idx=idx,
        config=config,
        logger=logger
    )

    adata = model.fit_likelihood()  
    return adata
