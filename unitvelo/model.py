import tensorflow as tf
import numpy as np
from scvelo.tools.utils import make_unique_list
from tqdm import tqdm
import scvelo as scv
from .optimize_utils import Model_Utils
from .utils import min_max
from packaging import version
import os

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
        Ms,
        Mu,
        var_names,
        idx=None,
        config=None
    ):
        super().__init__(
            adata=adata, 
            var_names=var_names,
            Ms=Ms,
            Mu=Mu,
            config = config
        )

        self.idx = idx
        self.scaling = adata.var['scaling'].values
        self.default_pars_names = ['gamma', 'beta', 'offset', 'a', 't', 'h', 'intercept']

        self.init_vars()
        self.init_weights()

        self.t_cell = self.compute_cell_time(args=None)
        self.pi = tf.constant(np.pi, dtype=tf.float32)

    def compute_cell_time(self, args=None, show=False):
        if args != None:
            x = self.init_time((0, 1), (3000, self.adata.n_vars))

            s_predict, u_predict = self.get_s_u(args, x)
            s_predict = tf.expand_dims(s_predict, axis=0) # 1 3000 d
            u_predict = tf.expand_dims(u_predict, axis=0)
            Mu = tf.expand_dims(self.Mu, axis=1) # n 1 d
            Ms = tf.expand_dims(self.Ms, axis=1)

            t_cell = self.match_time(Ms, Mu, s_predict, u_predict, x.numpy())

            if self.config.AGGREGATE_T:
                t_cell = tf.reshape(t_cell, (-1, 1))
                t_cell = tf.broadcast_to(t_cell, self.adata.shape)

        else:
            t_cell = self.init_time((0, 1), self.adata.shape)

            if self.config.IROOT == 'gcount':
                print ('---> Use Gene Counts as initial.')
                self.adata.obs['gcount'] = np.sum(self.adata.X.todense() > 0, axis=1)
                g_time = 1 - min_max(self.adata.obs.groupby(self.adata.uns['label'])['gcount'].mean())

                for id in list(g_time.index):
                    self.adata.obs.loc[self.adata.obs[self.adata.uns['label']] == id, 'gcount'] = g_time[id]

                scv.pl.scatter(self.adata, color='gcount', cmap='gnuplot', dpi=100)
                t_cell = tf.cast(
                    tf.broadcast_to(
                        self.adata.obs['gcount'].values.reshape(-1, 1), 
                        self.adata.shape), 
                    tf.float32)

            elif self.config.IROOT in self.adata.obs[self.adata.uns['label']].values:
                print ('Use diffusion pseudotime as initial.')
                import scanpy as sc
                sc.tl.diffmap(self.adata)
                self.adata.uns['iroot'] = \
                    np.flatnonzero(
                        self.adata.obs[self.adata.uns['label']] == self.config.IROOT
                    )[0]
                sc.tl.dpt(self.adata)

                if show:
                    scv.pl.scatter(self.adata, color='dpt_pseudotime', cmap='gnuplot', dpi=100)
                
                t_cell = tf.cast(
                    tf.broadcast_to(
                        self.adata.obs['dpt_pseudotime'].values.reshape(-1, 1), 
                        self.adata.shape), 
                    tf.float32)
        
            else:
                pass

        return t_cell

    def compute_loss(self, args, t_cell, iter, progress_bar):
        self.s_func, self.u_func = self.get_s_u(args, t_cell)
        udiff, sdiff = self.Mu - self.u_func, self.Ms - self.s_func

        self.u_r2 = square(udiff)
        self.s_r2 = square(sdiff)

        if (self.config.FIT_OPTION == '1') & \
            (iter > int(0.9 * self.config.MAX_ITER)) & self.config.REG_LOSS:
            self.s_r2 = self.s_r2 + \
                std(self.Ms, axis=0) * self.config.REG_TIMES * \
                exp(-square(args[4] - 0.5) / self.config.REG_SCALE)

        #! convert for self.varu to account for scaling in pre-processing
        self.vars = mean(self.s_r2, axis=0) \
            - square(mean(tf.math.sign(sdiff) * sqrt(self.s_r2), axis=0))
        self.varu = mean(self.u_r2 * square(self.scaling), axis=0) \
            - square(mean(tf.math.sign(udiff) * sqrt(self.u_r2) * self.scaling, axis=0))

        self.u_log_likeli = \
            - (self.Mu.shape[0] / 2) * log(2 * self.pi * self.varu) \
            - sum(self.u_r2 * square(self.scaling), axis=0) / (2 * self.varu) 
        self.s_log_likeli = \
            - (self.Ms.shape[0] / 2) * log(2 * self.pi * self.vars) \
            - sum(self.s_r2, axis=0) / (2 * self.vars) 

        error_1 = np.sum(sum(self.u_r2, axis=0).numpy()[self.idx]) / np.sum(self.idx)
        error_2 = np.sum(sum(self.s_r2, axis=0).numpy()[self.idx]) / np.sum(self.idx)
        progress_bar.set_description(f'Loss (Total): {(error_1 + error_2):.3f}, (Spliced): {error_2:.3f}, (Unspliced): {error_1:.3f}')
        
        return self.get_loss(iter,
                            sum(self.s_r2, axis=0), 
                            sum(self.u_r2, axis=0))

    def fit_likelihood(self):
        os.environ['TF_USE_LEGACY_KERAS'] = '1'
        if version.parse(tf.__version__) >= version.parse('2.11.0'):
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.config.LEARNING_RATE, amsgrad=True)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE, amsgrad=True)
        
        pre = tf.repeat(1e6, self.Ms.shape[1]) # (2000, )

        progress_bar = tqdm(range(self.config.MAX_ITER))
        for iter in progress_bar:
            with tf.GradientTape() as tape:                
                args = [
                    self.log_gamma, 
                    self.log_beta, 
                    self.offset, 
                    self.log_a, 
                    self.t, 
                    self.log_h, 
                    self.intercept
                ]
                obj = self.compute_loss(args, self.t_cell, iter, progress_bar)

            stop_cond = self.get_stop_cond(iter, pre, obj)

            if iter == self.config.MAX_ITER - 1 or tf.math.reduce_all(stop_cond) == True:
                self.t_cell = self.compute_cell_time(args=args)
                _ = self.get_fit_s(args, self.t_cell)
                s_derivative = self.get_s_deri(args, self.t_cell)

                self.post_utils(args)
                break

            args_to_optimize = self.get_opt_args(iter, args)
            gradients = tape.gradient(target=obj, sources=args_to_optimize)

            # convert gradients of variables with unused genes to 0
            # keep other gradients by multiplying 1
            convert = tf.cast(self.idx, tf.float32)
            processed_grads = [g * convert for g in gradients]

            optimizer.apply_gradients(zip(processed_grads, args_to_optimize))
            pre = obj

            if iter > 0 and int(iter % 800) == 0:
                self.t_cell = self.compute_cell_time(args=args)

        self.adata.layers['fit_t'] = self.t_cell.numpy() if self.config.AGGREGATE_T else self.t_cell
        self.adata.var['velocity_genes'] = self.idx
        self.adata.layers['fit_t'][:, ~self.adata.var['velocity_genes'].values] = np.nan

        return self.get_interim_t(self.t_cell, self.adata.var['velocity_genes'].values), s_derivative.numpy(), self.adata

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

        r2_spliced = 1 - sum(self.s_r2, axis=0) / var(self.Ms, axis=0) \
            / (self.adata.shape[0] - 1)
        r2_unspliced = 1 - sum(self.u_r2, axis=0) / var(self.Mu, axis=0) \
            / (self.adata.shape[0] - 1)

        self.adata.var['fit_sr2'] = r2_spliced.numpy()
        self.adata.var['fit_ur2'] = r2_unspliced.numpy()

    def save_pars(self, paras):
        columns = ['a', 'h', 'gamma', 'beta']
        for i, name in enumerate(self.default_pars_names):
            self.adata.var[f"fit_{name}"] = np.transpose(np.squeeze(paras[i]))

            if name in columns:
                self.adata.var[f"fit_{name}"] = np.exp(self.adata.var[f"fit_{name}"])            

def lagrange(
    adata,
    idx=None,
    Ms=None,
    Mu=None,
    var_names="velocity_genes",
    config=None
):
    if len(set(adata.var_names)) != len(adata.var_names):
        adata.var_names_make_unique()

    var_names = adata.var_names[idx]
    var_names = make_unique_list(var_names, allow_array=True)

    model = Recover_Paras(
        adata,
        Ms,
        Mu,
        var_names,
        idx=idx,
        config=config
    )

    latent_time_gm, s_derivative, adata = model.fit_likelihood()

    if 'latent_time' in adata.obs.columns:
        del adata.obs['latent_time']
    adata.obs['latent_time_gm'] = min_max(latent_time_gm[:, 0])

    return s_derivative, adata
